#!/usr/bin/env python3
"""Vector architecture PDF and audited layer-by-layer XLSX for current RTL model."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import sys,json,hashlib,math
sys.path.insert(0,str(ROOT/'models'))
import numpy as np
import torch
from torch import nn
from advanced_model import AdvancedDetector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,PatternFill,Alignment,Border,Side
from openpyxl.worksheet.table import Table,TableStyleInfo
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.page import PageMargins
from openpyxl.drawing.image import Image as XLImage

OUT=ROOT/'report/network_structure';OUT.mkdir(parents=True,exist_ok=True)
M=json.loads((ROOT/'rtl/generated/manifest.json').read_text())
font_path='/usr/share/fonts/google-droid/DroidSansFallback.ttf'
font_manager.fontManager.addfont(font_path)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font_path).get_name(),
                     'axes.unicode_minus':False,'pdf.fonttype':42,'font.size':10})
torch.set_num_threads(4)
original=AdvancedDetector().eval()
train_params=sum(p.numel() for p in original.parameters())
model=original.fused()
state=torch.load(ROOT/M['checkpoint'],map_location='cpu',weights_only=False)['state_dict']
assert 'c1.0.bias' in state,'Expected current BN-folded checkpoint'
model.load_state_dict(state)
seen={};hooks=[]
for name,module in model.named_modules():
    if isinstance(module,(nn.Conv2d,nn.Upsample)):
        hooks.append(module.register_forward_hook(lambda mod,ins,out,n=name:seen.update({n:(list(ins[0].shape),list(out.shape))})))
with torch.inference_mode():result=model(torch.zeros(1,1,96,96))
for h in hooks:h.remove()
assert list(result.shape)==[1,6,24,24]
conv=[]
for i,l in enumerate(M['layers']):
    d=l['descriptor'];name=l['name'];key=name+'.0' if i<9 else name
    mod=model.get_submodule(key);a,b=seen[key]
    assert b[1:]==l['shape']
    assert a==[1,d[7]+d[8],d[6],d[5]]
    assert mod.weight.numel()==d[11]*(d[7]+d[8])*d[12]**2
    assert mod.bias.numel()==d[11]
    conv.append(dict(index=i,name=name,op='Conv2D + ReLU + Requant' if d[15] else 'Conv2D + Requant (logits)',
                     input_shape=a,output_shape=b,kernel=list(mod.kernel_size),stride=list(mod.stride),
                     padding=list(mod.padding),weights=mod.weight.numel(),bias=mod.bias.numel(),
                     weight_shape=list(mod.weight.shape),macs=l['macs'],relu=bool(d[15]),descriptor=d,
                     input_scale=l['input_scale'],output_scale=l['output_scale']))
assert sum(r['weights'] for r in conv)==89820
assert sum(r['bias'] for r in conv)==274
assert sum(r['macs'] for r in conv)==20321280
assert sum(p.numel() for p in model.parameters())==90094
assert train_params==90362
C={r['name']:r for r in conv}
source={
'c1':['input'],'c2':['c1'],'c3':['c2'],'c4':['c3'],'context':['c4'],
'fuse1':['concat1'],'fuse2':['concat2'],'head':['fuse2'],'regression':['head'],
'heatmap':['head'],'offset':['regression'],'size':['regression'],'quality':['regression']}
def num(s):return math.prod(s)
def sh(s):return ' × '.join(map(str,s))
ops=[]
def add(name,kind,sources,ins,out,kernel='—',stride='—',weights=0,bias=0,note=''):
    ops.append(dict(name=name,op=kind,sources=sources,inputs=ins,output=out,kernel=kernel,stride=stride,
                    weights=weights,bias=bias,note=note))
add('input','Input',[],[],[1,1,96,96],note='입력 q=(gray*127+127)//255, scale=1/127')
for name in ['c1','c2','c3','c4','context']:
    r=C[name];add(name,r['op'],source[name],[r['input_shape']],r['output_shape'],sh(r['kernel']),sh(r['stride']),r['weights'],r['bias'],'BN folding; padding=1')
add('up1','Nearest upsample ×2',['context'],[[1,48,6,6]],[1,48,12,12],note='파라미터 없음; RTL fuse1 주소 발생기로 구현')
add('concat1','Concat channels',['up1','c3'],[[1,48,12,12],[1,32,12,12]],[1,80,12,12],note='RTL 가상 concat; 별도 tensor SRAM 미할당')
r=C['fuse1'];add('fuse1',r['op'],source['fuse1'],[r['input_shape']],r['output_shape'],'3 × 3','1 × 1',r['weights'],r['bias'],'BN folding; padding=1')
add('up2','Nearest upsample ×2',['fuse1'],[[1,32,12,12]],[1,32,24,24],note='파라미터 없음; RTL fuse2 주소 발생기로 구현')
add('concat2','Concat channels',['up2','c2'],[[1,32,24,24],[1,24,24,24]],[1,56,24,24],note='RTL 가상 concat; 별도 tensor SRAM 미할당')
for name in ['fuse2','head','regression','heatmap','offset','size','quality']:
    r=C[name];add(name,r['op'],source[name],[r['input_shape']],r['output_shape'],sh(r['kernel']),sh(r['stride']),r['weights'],r['bias'],
                  'BN folding; padding=1' if r['relu'] else 'padding=0; activation 없음; 후처리에서 sigmoid 적용')
add('output','Concat channels',['heatmap','offset','size','quality'],[[1,1,24,24],[1,2,24,24],[1,2,24,24],[1,1,24,24]],[1,6,24,24],note='최종 logits 패킹; 순서 heatmap,offset_x,offset_y,width,height,quality')
assert len(ops)==19
assert sum(o['weights'] for o in ops)==89820 and sum(o['bias'] for o in ops)==274

# ----- PDF/vector drawing -----
blue='#DBEAFE';orange='#FFEDD5';teal='#CCFBF1';gray='#F1F5F9';green='#DCFCE7';ink='#16324F'
fig=plt.figure(figsize=(16.54,11.69),facecolor='white')
ax=fig.add_axes([.025,.07,.95,.82]);ax.set(xlim=(0,17),ylim=(0,12));ax.axis('off')
fig.text(.04,.957,'96×96 사람 검출 네트워크',fontsize=23,color=ink,weight='bold')
fig.text(.04,.924,'student_kd  |  13 Conv / 2 Upsample / 3 Concat  |  BN folding 후 90,094 parameters  |  20.321M MACs/frame',fontsize=12,color='#486581')
pos={}
for n,x in zip(['input','c1','c2','c3','c4','context'],[1.4,4.2,7,9.8,12.6,15.4]):pos[n]=(x,10.2)
for n,x in zip(['fuse2','concat2','up2','fuse1','concat1','up1'],[1.4,4.2,7,9.8,12.6,15.4]):pos[n]=(x,7.2)
pos.update(head=(1.4,4.8),regression=(7,4.8),heatmap=(1.4,2.4),offset=(7,2.4),size=(9.8,2.4),quality=(12.6,2.4),output=(15.4,2.4))
W,H=2.15,1.25
for o in ops:
    name=o['name'];x,y=pos[name]
    color=blue if name in C and C[name]['relu'] else orange if name in C else teal if name.startswith('up') else green if name=='output' else gray
    ax.add_patch(FancyBboxPatch((x-W/2,y-H/2),W,H,boxstyle='round,pad=0.025,rounding_size=0.12',fc=color,ec=ink,lw=1.2,zorder=3))
    if name in C:
        r=C[name];lab=f"Conv {r['kernel'][0]}×{r['kernel'][1]} / s{r['stride'][0]}"+(' + ReLU' if r['relu'] else ' / logits')
    else:lab='Nearest ×2' if name.startswith('up') else 'Channel concat' if name!='input' else 'Grayscale / INT8'
    ax.text(x,y+.36,name,ha='center',va='center',fontsize=12,weight='bold',color=ink,zorder=4)
    ax.text(x,y+.02,lab,ha='center',va='center',fontsize=8.5,zorder=4)
    ax.text(x,y-.34,sh(o['output'][1:])+f"  ({num(o['output']):,})",ha='center',va='center',fontsize=8.5,zorder=4)
def arrow(a,b,col=ink,lw=1.4):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color=col,lw=lw,zorder=2))
def path(points,col=ink,arrowlast=True,lw=1.4):
    for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=col,lw=lw,zorder=2)
    if arrowlast:arrow(points[-2],points[-1],col,lw)
    else:ax.plot([points[-2][0],points[-1][0]],[points[-2][1],points[-1][1]],color=col,lw=lw,zorder=2)
for a,b in zip(['input','c1','c2','c3','c4'],['c1','c2','c3','c4','context']):arrow((pos[a][0]+W/2,10.2),(pos[b][0]-W/2,10.2))
arrow((15.4,10.2-H/2),(15.4,7.2+H/2))
for a,b in zip(['up1','concat1','fuse1','up2','concat2'],['concat1','fuse1','up2','concat2','fuse2']):arrow((pos[a][0]-W/2,7.2),(pos[b][0]+W/2,7.2))
path([(9.8,10.2-H/2),(9.8,8.95),(12.6,8.95),(12.6,7.2+H/2)],'#7C3AED',lw=2)
ax.text(11.2,9.08,'c3 skip: 32×12×12',ha='center',fontsize=9,color='#7C3AED')
path([(7,10.2-H/2),(7,8.5),(4.2,8.5),(4.2,7.2+H/2)],'#7C3AED',lw=2)
ax.text(5.6,8.64,'c2 skip: 24×24×24',ha='center',fontsize=9,color='#7C3AED')
arrow((1.4,7.2-H/2),(1.4,4.8+H/2))
arrow((1.4+W/2,4.8),(7-W/2,4.8))
ax.text(4.2,5.03,'共有 feature: 24×24×24'.replace('共有','공유'),ha='center',fontsize=9,color='#486581')
arrow((1.4,4.8-H/2),(1.4,2.4+H/2))
ax.plot([7,7],[4.8-H/2,3.55],color=ink,lw=1.4,zorder=2)
ax.plot([7,12.6],[3.55,3.55],color=ink,lw=1.4,zorder=2)
for x in [7,9.8,12.6]:arrow((x,3.55),(x,2.4+H/2))
for x in [1.4,7,9.8,12.6]:ax.plot([x,x],[2.4-H/2,.72],color='#0F766E',lw=1.5,zorder=2)
ax.plot([1.4,15.4],[.72,.72],color='#0F766E',lw=1.5,zorder=2)
arrow((15.4,.72),(15.4,2.4-H/2),'#0F766E',1.5)
ax.text(8.5,.3,'Channel order: heatmap, offset_x, offset_y, width, height, quality',ha='center',fontsize=10,color='#0F766E')
ax.text(.3,11.35,'BACKBONE',fontsize=12,weight='bold',color='#486581')
ax.text(3.1,6.1,'DECODER / SKIP FUSION',fontsize=10,weight='bold',color='#486581')
ax.text(9.8,5.25,'DETECTION HEADS',fontsize=10,weight='bold',color='#486581')
fig.text(.04,.038,'Shape = C×H×W, batch N=1; 괄호 안 숫자 = 원소 수. 파란색: 3×3 Conv+ReLU / 주황색: 1×1 logits head / 보라색: skip.',fontsize=10,color=ink)
fig.text(.04,.016,'RTL은 upsample/중간 concat을 주소 발생기로 처리한다. 후처리 sigmoid·좌표 decode·threshold·NMS는 학습 네트워크 밖이다.',fontsize=10,color=ink)
fig.savefig(OUT/'network_structure.png',dpi=160,facecolor='white')

fig2=plt.figure(figsize=(16.54,11.69),facecolor='white')
fig2.text(.035,.95,'Convolution 레이어 상세 — BN folding 후 RTL 추론 기준',fontsize=20,color=ink,weight='bold')
fig2.text(.035,.914,'입출력 shape는 NCHW. Weight shape는 OIHW. Bias는 INT32이며 첫 9개 블록의 BN 효과를 합친 값이다.',fontsize=11,color='#486581')
a2=fig2.add_axes([.035,.30,.93,.57]);a2.axis('off')
headers=['Layer','Kernel / s / p','Input NCHW','Output NCHW','Weight','Bias','MACs']
rows=[]
for r in conv:rows.append([r['name'],f"{r['kernel'][0]}×{r['kernel'][1]} / {r['stride'][0]} / {r['padding'][0]}",sh(r['input_shape']),sh(r['output_shape']),f"{r['weights']:,}",str(r['bias']),f"{r['macs']:,}"])
rows.append(['합계','13 Conv','—','6×24×24 logits',f'{89820:,}','274',f'{20321280:,}'])
tab=a2.table(cellText=rows,colLabels=headers,loc='center',cellLoc='center',colWidths=[.11,.16,.20,.20,.11,.07,.15]);tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,2.25)
for (r,c),cell in tab.get_celld().items():
    cell.set_edgecolor('#D6E1EC');cell.set_linewidth(.6)
    if r==0:cell.set_facecolor(ink);cell.set_text_props(color='white',weight='bold')
    elif r==len(rows):cell.set_facecolor('#DBEAFE');cell.set_text_props(weight='bold')
    else:cell.set_facecolor('#F2F6FA' if r%2==0 else 'white')
notes=[
'모델 파라미터: Weight 89,820 + Bias 274 = 90,094개. 원본 학습형(BN 포함)의 trainable parameter는 90,362개다.',
'입력: 1×1×96×96 grayscale INT8. 출력: 1×6×24×24 INT8 logits (3,456개). 1 MAC은 multiply-accumulate 1회다.',
'INT8 weight/activation, INT32 bias/누산, INT64 재양자화 중간값. MACs는 padding 위치를 포함한 명목 연산 수다.',
'현재 SRAM: input CDC 9,216 + activation 108,288 + weight 89,820 + bias 1,096 + multiplier 1,096 + descriptor 832 + post 9,984 = 220,332 bytes.',
'현재 activation은 레이어별 주소를 분할한 큰 SRAM 하나다. 두 BANK 재사용 최적화는 아직 적용하지 않았다.',
'Clock: DVP/render 27MHz, NPU/post 100MHz. 코어 40,965,537 clocks ≈409.655ms, 코어 처리율 상한 약2.441 FPS.',
'상세 테이블·전체 연산·채널별 scale·SRAM 주소·출력 해석은 함께 제공한 Excel 파일에 있다.'
]
for i,line in enumerate(notes):fig2.text(.04,.255-i*.032,line,fontsize=9.5,color=ink)
with PdfPages(OUT/'person_detector_network.pdf') as pdf:
    pdf.infodict().update(Title='96x96 Person Detector Architecture',Author='Codex',Subject='Current BN-folded INT8 RTL model, dual-clock pixel27MHz / NPU100MHz')
    pdf.savefig(fig);pdf.savefig(fig2)
plt.close(fig);plt.close(fig2)

# ----- Excel: numeric tables, filters, frozen panes, printed headers -----
wb=Workbook();summary=wb.active;summary.title='요약'
navy='16324F';pale='E8F0FA';white='FFFFFF';accent='0F766E'
headfill=PatternFill('solid',fgColor=navy);thin=Side(style='thin',color='D6E1EC')
def sheet(name,title,description,headers,rows,widths):
    ws=wb.create_sheet(name);n=len(headers)
    ws.merge_cells(start_row=1,start_column=1,end_row=1,end_column=n);ws.cell(1,1,title).font=Font(name='맑은 고딕',size=16,bold=True,color=navy)
    ws.merge_cells(start_row=2,start_column=1,end_row=2,end_column=n);ws.cell(2,1,description).alignment=Alignment(wrap_text=True,vertical='center');ws.row_dimensions[2].height=36
    for j,h in enumerate(headers,1):ws.cell(4,j,h)
    for row in rows:ws.append(row)
    for cell in ws[4]:cell.font=Font(name='맑은 고딕',bold=True,color=white);cell.fill=headfill;cell.alignment=Alignment(wrap_text=True,vertical='center')
    ws.row_dimensions[4].height=34
    for row in ws.iter_rows(min_row=5):
        for c in row:
            c.font=Font(name='맑은 고딕',size=10);c.alignment=Alignment(vertical='center',wrap_text=True);c.border=Border(bottom=thin)
            if isinstance(c.value,(int,float)):c.number_format='#,##0' if isinstance(c.value,int) else '0.0000000000'
        ws.row_dimensions[row[0].row].height=36
    if rows:
        table=Table(displayName=f'Table{len(wb.worksheets)}',ref=f'A4:{get_column_letter(n)}{4+len(rows)}');table.tableStyleInfo=TableStyleInfo(name='TableStyleMedium2',showRowStripes=True);ws.add_table(table)
    for i,w in enumerate(widths,1):ws.column_dimensions[get_column_letter(i)].width=w
    ws.freeze_panes='C5';ws.sheet_view.zoomScale=80
    ws.sheet_properties.pageSetUpPr.fitToPage=True
    ws.page_setup.orientation='landscape';ws.page_setup.paperSize=ws.PAPERSIZE_A3;ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=0
    ws.print_title_rows='1:4';ws.print_area=f'A1:{get_column_letter(n)}{4+len(rows)}';ws.page_margins=PageMargins(left=.25,right=.25,top=.4,bottom=.4)
    return ws
summary.merge_cells('A1:D1');summary['A1']='96×96 사람 검출 네트워크 — 현재 RTL 기준';summary['A1'].font=Font(name='맑은 고딕',size=18,bold=True,color=navy)
info=[
('기준','student_kd / BatchNorm folding 후 INT8 추론','출처','models/advanced_model.py + rtl/generated/manifest.json'),
('입력 shape (NCHW)','1 × 1 × 96 × 96','입력 원소 수',9216),
('출력 shape (NCHW)','1 × 6 × 24 × 24','출력 원소 수',3456),
('Convolution 수',13,'Upsample / Concat 수','2 / 3'),
('INT8 Weight 수',89820,'INT32 Bias 수',274),
('모델 parameter 합계',90094,'모델 parameter 저장량 bytes',90916),
('원본 학습 모델 trainable 수',train_params,'학습형 BN gamma+beta 수',536),
('MACs / frame',20321280,'MAC 정의','multiply-accumulate 1회; 명목 padding 포함'),
('전체 SRAM bytes',220332,'KiB',220332/1024),
('Activation SRAM bytes',108288,'현재 구현','주소 구간 분할 단일 SRAM; 2BANK 최적화 미적용'),
('Input CDC SRAM bytes',9216,'입력 양자화','q=(gray*127+127)//255; scale=1/127; zero point=0'),
('Pixel clock Hz',27000000,'NPU clock Hz',100000000),
('NPU clocks / frame',40965537,'코어 시간 ms',409.65537),
('코어 최대 inference FPS',1e8/40965537,'주의','캡처·후처리·VSYNC 대기 추가; FPGA timing 미검증'),
('Multiplier/descriptor/LUT 개수','274 / 208 words / 1536','비학습 실행 데이터','모델 Weight/Bias 합계와 별도'),
('Shape 표기','NCHW, batch N=1','Weight shape','OIHW = Cout × Cin × Kh × Kw'),
('요소 수 의미','채널·공간 전체 scalar 개수','합산 금지','레이어 output 합계는 peak SRAM/live memory와 다름'),
('가상 중간 tensor','up1/concat1/up2/concat2','RTL 실현','주소 매핑으로 read; 별도 메모리 할당 없음'),
('Bias 정의','첫 9개 Conv+BN의 BN을 접은 INT32 bias','원본 학습형','첫 9개 Conv bias 없음; BN gamma/beta 별도'),
('고정 모델 SHA256',M['checkpoint_sha256'],'학습 checkpoint',M['checkpoint']),
]
for r,row in enumerate(info,4):
    for c,v in enumerate(row,1):
        cell=summary.cell(r,c,v);cell.font=Font(name='맑은 고딕',size=11,bold=c in [1,3],color=navy);cell.alignment=Alignment(vertical='center',wrap_text=True);cell.fill=PatternFill('solid',fgColor=pale if r%2==0 else white)
        if isinstance(v,(int,float)):cell.number_format='#,##0' if isinstance(v,int) else '0.0000'
    summary.row_dimensions[r].height=34
for c,w in zip('ABCD',[30,61,31,73]):summary.column_dimensions[c].width=w
summary.freeze_panes='B4';summary.sheet_view.zoomScale=90;summary.page_setup.orientation='landscape';summary.page_setup.paperSize=summary.PAPERSIZE_A3;summary.sheet_properties.pageSetUpPr.fitToPage=True;summary.page_setup.fitToWidth=1

h=['순서','레이어','연산','입력 source','입력 shape NCHW','Cin','Hin','Win','입력 원소 수','Kernel H×W','Stride H×W','Padding H×W','출력 shape NCHW','Cout','Hout','Wout','출력 원소 수','Weight shape OIHW','Weight 수','Bias 수','총 parameter 수','MACs','ReLU','출력 INT8 bytes','Weight INT8 bytes','Bias INT32 bytes']
rows=[]
for r in conv:
 a,b=r['input_shape'],r['output_shape'];rows.append([r['index'],r['name'],r['op'],', '.join(source[r['name']]),sh(a),*a[1:],num(a),sh(r['kernel']),sh(r['stride']),sh(r['padding']),sh(b),*b[1:],num(b),sh(r['weight_shape']),r['weights'],r['bias'],r['weights']+r['bias'],r['macs'],'Yes' if r['relu'] else 'No',num(b),r['weights'],4*r['bias']])
ws=sheet('Conv_13layers','Convolution 레이어 — BN folding 후','입출력 tensor는 NCHW, batch=1. 숫자 열은 Excel 숫자다. first 9 blocks의 BN/ReLU는 Conv에 접히거나 포함되어 있으며 head4개는 logits다.',h,rows,[9,15,32,20,26,10,10,10,17,15,15,16,26,10,10,10,17,26,16,15,18,18,12,20,20,20])
total=19
for col in [19,20,21,22,25,26]:ws.cell(total,col,f'=SUM({get_column_letter(col)}5:{get_column_letter(col)}17)')
ws.cell(total,2,'총계 (13 Conv)');ws.cell(total,2).font=Font(bold=True,color=navy);ws.cell(total,5,'출력 원소 수는 레이어별 관찰값; 합산해 SRAM으로 쓰지 않음')
for cell in ws[total]:cell.fill=PatternFill('solid',fgColor=pale)
ws.print_area=f'A1:Z{total}'
rows=[]
for i,o in enumerate(ops):rows.append([i,o['name'],o['op'],', '.join(o['sources']) or '외부 입력',' + '.join(sh(a) for a in o['inputs']) or '—',sum(num(a) for a in o['inputs']),sh(o['output']),num(o['output']),o['kernel'],o['stride'],o['weights'],o['bias'],o['weights']+o['bias'],o['note']])
sheet('전체연산','전체 논리 연산 — Input·Upsample·Concat 포함','19행 = Input1 + Conv13 + Upsample2 + Concat3. 중간 upsample/concat은 논리 shape를 표시한 것이며 현재 RTL에서 별도 SRAM tensor로 저장하지 않는다.',
      ['순서','연산 이름','종류','입력 source','입력 shape(s) NCHW','입력 원소 수 합','출력 shape NCHW','출력 원소 수','Kernel','Stride','Weight 수','Bias 수','Parameter 수','비고'],rows,[9,16,34,34,54,20,27,20,15,15,17,15,18,75])
memories=[('input_frame_mem','input CDC buffer',9216,8,'pixel write27 / NPU read100','입력 frame 완성 후 소유권 고정'),('activation_mem','activation + input copy + output',108288,8,'NPU100','단일 큰 SRAM, 레이어별 주소 구간'),('weight_mem','INT8 model weight',89820,8,'NPU100','preload space0'),('bias_mem','INT32 model bias',274,32,'NPU100','preload space1'),('multiplier_mem','INT32 requant multiplier',274,32,'NPU100','preload space3'),('descriptor_mem','layer metadata',208,32,'NPU100','preload space2'),('lut_mem','sigmoid / quality sqrt',1536,16,'NPU100','preload space4'),('result_mem','post probability',3456,16,'NPU100','logits가 아니라 Q0.16 확률')]
ws=sheet('SRAM_주소','현재 SRAM 용량과 구조','논리 용량이다. Vivado BRAM 단위 반올림과 약1.8KB 후보/박스 register는 별도다. 두 BANK 재사용은 아직 미적용이다.',
         ['메모리','내용','Depth','Width bits','용량 bytes','KiB','Clock','비고'],[[n,d,count,bits,count*bits//8,count*bits/8192,clk,note] for n,d,count,bits,clk,note in memories],[26,40,17,17,20,18,36,68])
ws.cell(14,1,'총 SRAM');ws.cell(14,5,'=SUM(E5:E12)');ws.cell(14,6,'=SUM(F5:F12)')
ws2=sheet('레이어주소_scale','레이어 주소와 scale','주소는 element index. activation은8bit element, descriptor/bias/multiplier는32bit word다. scale은 host 메타데이터이며 HW는 multiplier와 fixed shift24를 사용한다.',
          ['Layer index','레이어','입력1 base','입력2 base','출력 base','출력 마지막','Weight base','Weight 마지막','Bias/Mult base','Bias/Mult 마지막','Desc 시작','Desc 끝','Input scale','Output scale','Upsample 첫 입력','ReLU'],
          [[r['index'],r['name'],r['descriptor'][0],r['descriptor'][1],r['descriptor'][2],r['descriptor'][2]+num(r['output_shape'])-1,r['descriptor'][3],r['descriptor'][3]+r['weights']-1,r['descriptor'][4],r['descriptor'][4]+r['bias']-1,r['index']*16,r['index']*16+15,r['input_scale'],r['output_scale'],bool(r['descriptor'][14]),r['relu']] for r in conv],
          [14,16,17,17,17,19,19,20,20,20,17,17,24,24,22,14])
def signedhex(path,bits):
 a=np.array([int(v,16) for v in path.read_text().split()],dtype=np.int64);return np.where(a>=(1<<(bits-1)),a-(1<<bits),a)
biasq=signedhex(ROOT/'rtl/generated/bias.hex',32);multq=signedhex(ROOT/'rtl/generated/requant.hex',32)
qrows=[]
for r,l in zip(conv,M['layers']):
 for oc,scale in enumerate(l['weight_scales']):
  idx=r['descriptor'][4]+oc
  qrows.append([r['name'],oc,idx,r['input_scale'],scale,r['input_scale']*scale,r['output_scale'],int(biasq[idx]),int(multq[idx]),24])
assert len(qrows)==274
sheet('채널별_양자화','채널별 weight·bias scale과 실제 계수','INT8 weight는 output channel별 scale. INT32 bias scale=input_scale×weight_scale. Bias_q와 multiplier는 실제 preload hex에서 읽었다. shift는공통24다.',
      ['레이어','Output channel','Bias/Mult index','Input scale','Weight scale','Bias scale','Output scale','Bias_q INT32','Multiplier INT32','Shift'],qrows,[17,21,22,25,25,25,25,24,24,14])
output_rows=[['0','heatmap','1×1×24×24',576,'0~575','sigmoid(z)','검출 score의 heat probability'],['1','offset_x','1×1×24×24',576,'576~1151','2*sigmoid(z)-0.5','center_x=(grid_x+offset_x)*4'],['2','offset_y','1×1×24×24',576,'1152~1727','2*sigmoid(z)-0.5','center_y=(grid_y+offset_y)*4'],['3','width','1×1×24×24',576,'1728~2303','sigmoid(z)*96','96좌표의 box width'],['4','height','1×1×24×24',576,'2304~2879','sigmoid(z)*96','96좌표의 box height'],['5','quality','1×1×24×24',576,'2880~3455','sqrt(sigmoid(z))','score=heat_prob*quality_sqrt']]
sheet('출력해석','최종6채널 logits와 RTL 후처리','학습 모델 출력은 logits다. sigmoid/좌표 변환/top50/NMS는 postprocessor에서 실행된다. 최종 bbox는96×96 inclusive 좌표, 최대50개이며 score는 Q0.16이다.',
      ['출력 채널','의미','Shape NCHW','원소 수','Stream index','후처리','설명'],output_rows,[17,20,27,18,23,31,65])
ws=wb.create_sheet('구조도');ws['A1']='논리 네트워크 구조 (PDF 1페이지와 동일)';ws['A1'].font=Font(name='맑은 고딕',size=18,bold=True,color=navy)
img=XLImage(str(OUT/'network_structure.png'));img.width=1323;img.height=935;ws.add_image(img,'A3');ws.sheet_view.zoomScale=65
wb.calculation.fullCalcOnLoad=True
wb.save(OUT/'person_detector_layers.xlsx')
# Reopen and verify workbook data (numeric cells, model totals, all graph nodes).
check=load_workbook(OUT/'person_detector_layers.xlsx',data_only=False)
w=check['Conv_13layers'];assert sum(w.cell(i,19).value for i in range(5,18))==89820
assert sum(w.cell(i,20).value for i in range(5,18))==274
assert sum(w.cell(i,22).value for i in range(5,18))==20321280
assert check['전체연산'].max_row==23 and check['채널별_양자화'].max_row==278
assert sum(check['SRAM_주소'].cell(i,5).value for i in range(5,13))==220332
report=dict(model_parameters_fused=90094,weights=89820,bias=274,training_parameters_unfused=train_params,
            conv_layers=13,upsample_operations=2,concat_operations=3,macs=20321280,total_sram_bytes=220332,
            input_nchw=[1,1,96,96],output_nchw=[1,6,24,24],logical_operations=ops,convolutions=conv,
            checkpoint_sha256=hashlib.sha256((ROOT/M['checkpoint']).read_bytes()).hexdigest(),
            source_manifest_sha256=hashlib.sha256((ROOT/'rtl/generated/manifest.json').read_bytes()).hexdigest())
assert report['checkpoint_sha256']==M['checkpoint_sha256']
(OUT/'network_metadata.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PASS: PyTorch checkpoint, RTL descriptors, all shapes/counts, XLSX numeric totals and220332-byte SRAM audited.')
for f in ['person_detector_network.pdf','person_detector_layers.xlsx','network_structure.png','network_metadata.json']:
 print(OUT/f,(OUT/f).stat().st_size)
