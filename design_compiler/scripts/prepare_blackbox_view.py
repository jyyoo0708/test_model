#!/usr/bin/env python3
"""Reproducible memory-only extraction of the current production RTL."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'rtl';DST=ROOT/'design_compiler/blackbox_rtl';DST.mkdir(parents=True,exist_ok=True)
VERIFY=DST/'verify';VERIFY.mkdir(exist_ok=True)
memories=[]
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b,1)
def ram(name,depth,width,aw,wclk,rclk,we,wa,wd,re_,ra,rd,owner):
    memories.append(dict(instance=f'{owner}/u_mem_{name}',depth=depth,width=width,address_width=aw))
    base=f'mem_{name}'
    ports=[('clk_write','',wclk),('clk_read','',rclk),('i_write_enable','',we),('i_write_address',f'[{aw}-1:0]',wa),('i_write_data',f'[{width}-1:0]',wd),('i_read_enable','',re_),('i_read_address',f'[{aw}-1:0]',ra),('o_read_data',f'[{width}-1:0]',None)]
    lines=[]
    for p,w,e in ports:
        if p.startswith('clk'):continue
        lines.append(f'wire {w} {base}_{p};')
    for p,w,e in ports:
        if p.startswith('clk'):continue
        lines.append(f'assign {rd} = {base}_{p};' if e is None else f'assign {base}_{p} = {e};')
    lines += [f'npu_sram_bb #(.p_depth({depth}), .p_data_w({width}), .p_addr_w({aw})) u_mem_{name} (']
    lines += [f'    .{p}({e if p.startswith("clk") else base+"_"+p})'+(',' if i<len(ports)-1 else '') for i,(p,w,e) in enumerate(ports)]
    lines += [');']
    return '\n'.join(lines)+'\n\n'
manifest={'production_modified':False,'memories_excluded_bytes':220332,'sources':[]}
for f in ['person_detector_top.v','dvp_capture.v']:
    s=(SRC/f).read_text();(DST/f).write_text(s);manifest['sources'].append(dict(source='rtl/'+f,sha256=sha(s),generated=f,generated_sha256=sha(s),change='byte-identical copy'))
f='npu_core.v';orig=(SRC/f).read_text();s=orig
s,n=re.subn(r'\(\* ram_style = "block" \*\) reg[^\n]+\n','',s);assert n==6
for a,b in [('reg [31:0] descriptor_data_q;','wire [31:0] descriptor_data_q;'),('reg signed [7:0] activation_data_q, weight_data_q;','wire signed [7:0] activation_data_q, weight_data_q;'),('reg signed [31:0] bias_data_q, multiplier_data_q;','wire signed [31:0] bias_data_q, multiplier_data_q;'),('reg signed [7:0] input_frame_data_q;','wire signed [7:0] input_frame_data_q;')]:s=once(s,a,b)
start=s.index('// Pixel writes are governed');end=s.index('// Closely coupled counters',start)
removed=s[start:end]
r='// Memory instances are black boxes for DC; functional models are separate.\n'
r+=ram('input_frame','9216','8','14','clk_pixel','clk_npu',"i_pixel_we && (i_pixel_addr < 14'd9216)",'i_pixel_addr','i_pixel_data','state_q == p_input_request','input_copy_index_q','input_frame_data_q','u_npu')
r+=ram('activation','p_activation_depth','8','p_activation_aw','clk_npu','clk_npu','activation_write_enable','activation_write_addr','activation_write_data','activation_read_enable','activation_read_addr','activation_data_q','u_npu')
r+=ram('weight','p_weight_depth','8','p_weight_aw','clk_npu','clk_npu',"!o_busy && i_load_we && (i_load_space == 3'd0) && (i_load_addr < p_weight_depth)",'i_load_addr[p_weight_aw-1:0]','i_load_data[7:0]','state_q == p_mac_request','weight_addr[p_weight_aw-1:0]','weight_data_q','u_npu')
r+=ram('bias','p_parameter_depth','32','p_parameter_aw','clk_npu','clk_npu',"!o_busy && i_load_we && (i_load_space == 3'd1) && (i_load_addr < p_parameter_depth)",'i_load_addr[p_parameter_aw-1:0]','i_load_data','state_q == p_parameter_request','parameter_addr[p_parameter_aw-1:0]','bias_data_q','u_npu')
r+=ram('multiplier','p_parameter_depth','32','p_parameter_aw','clk_npu','clk_npu',"!o_busy && i_load_we && (i_load_space == 3'd3) && (i_load_addr < p_parameter_depth)",'i_load_addr[p_parameter_aw-1:0]','i_load_data','state_q == p_parameter_request','parameter_addr[p_parameter_aw-1:0]','multiplier_data_q','u_npu')
r+=ram('descriptor','p_descriptor_depth','32','p_descriptor_aw','clk_npu','clk_npu',"!o_busy && i_load_we && (i_load_space == 3'd2) && (i_load_addr < p_descriptor_depth)",'i_load_addr[p_descriptor_aw-1:0]','i_load_data','state_q == p_desc_request','descriptor_addr[p_descriptor_aw-1:0]','descriptor_data_q','u_npu')
r+='always @(posedge clk_npu) begin\n    if (state_q == p_mac_request) begin\n        activation_padding_q <= padding_pixel;\n    end\nend\n\n'
s=s[:start]+r+s[end:];(DST/f).write_text(s);manifest['sources'].append(dict(source='rtl/'+f,sha256=sha(orig),generated=f,generated_sha256=sha(s),change='six memory declarations/data nets and memory access region only',removed_memory_region_sha256=sha(removed)))
f='bbox_overlay.v';orig=(SRC/f).read_text();s=orig
s,n=re.subn(r'\(\* ram_style = "block" \*\) reg[^\n]+\n','',s);assert n==2
s=once(s,'reg [15:0] lut_data_q;','wire [15:0] lut_data_q;');s=once(s,'reg [15:0] result_read_q;','wire [15:0] result_read_q;')
start=s.index('// Both large memories expose');end=s.index('always @(posedge clk_npu or negedge rst_npu_n)',start)
removed=s[start:end]
r='// Memory instances are black boxes for DC; functional models are separate.\n'
r+=ram('lut','1536','16','11','clk_npu','clk_npu',"i_lut_we && (i_lut_addr < 11'd1536)",'i_lut_addr','i_lut_data',"i_result_valid && (i_result_index < 12'd3456)",'lut_read_addr','lut_data_q','u_post')
r+=ram('result','3456','16','12','clk_npu','clk_npu','lut_valid_q','lut_index_q','lut_data_q',"1'b1",'result_read_addr','result_read_q','u_post')
s=s[:start]+r+s[end:];(DST/f).write_text(s);manifest['sources'].append(dict(source='rtl/'+f,sha256=sha(orig),generated=f,generated_sha256=sha(s),change='two memory declarations/data nets and memory access region only',removed_memory_region_sha256=sha(removed)))
ports='''module npu_sram_bb #(
    parameter integer p_depth = 1024,
    parameter integer p_data_w = 8,
    parameter integer p_addr_w = 10
) (
    input wire clk_write,
    input wire clk_read,
    input wire i_write_enable,
    input wire [p_addr_w-1:0] i_write_address,
    input wire [p_data_w-1:0] i_write_data,
    input wire i_read_enable,
    input wire [p_addr_w-1:0] i_read_address,
'''
(DST/'npu_sram_bb.v').write_text('`default_nettype none\n(* black_box = "true", syn_black_box = 1 *)\n'+ports+'''    output wire [p_data_w-1:0] o_read_data
);
// Intentionally no data/timing/area model. Preserve all eight instances in DC.
endmodule
`default_nettype wire
''')
(VERIFY/'npu_sram_bb.v').write_text('`default_nettype none\n'+ports+'''    output reg [p_data_w-1:0] o_read_data
);
reg [p_data_w-1:0] mem [0:p_depth-1];
always @(posedge clk_write) begin
    if (i_write_enable) mem[i_write_address] <= i_write_data;
end
always @(posedge clk_read) begin
    if (i_read_enable) o_read_data <= mem[i_read_address];
end
endmodule
`default_nettype wire
''')
# Only hierarchy references change in the existing driver; all checks remain.
s=(SRC/'tb/test_pipeline.cpp').read_text()
for name in ['input_frame','activation','weight','bias','multiplier','descriptor']:
    a=f'person_detector_top__DOT__u_npu__DOT__{name}_mem'
    b=f'person_detector_top__DOT__u_npu__DOT__u_mem_{name}__DOT__mem'
    assert a in s;s=s.replace(a,b)
(VERIFY/'test_pipeline_blackbox_view.cpp').write_text(s)
defaults={'p_activation_depth':108288,'p_weight_depth':89820,'p_parameter_depth':274,'p_descriptor_depth':208,'p_activation_aw':17,'p_weight_aw':17,'p_parameter_aw':9,'p_descriptor_aw':8}
for memory in memories:
    depth=memory['depth'];width=memory['width'];aw=memory['address_width']
    memory['default_depth']=int(depth) if depth.isdigit() else defaults[depth]
    memory['default_width']=int(width)
    memory['default_address_width']=int(aw) if aw.isdigit() else defaults[aw]
    memory['logical_bytes']=memory['default_depth']*memory['default_width']//8
assert sum(x['logical_bytes'] for x in memories)==220332
manifest['memories']=memories;assert len(memories)==8
manifest['instance_pattern']='*/u_mem_*';manifest['reference_pattern']='npu_sram_bb*'
(DST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(DST/'sources.f').write_text('\n'.join(str((DST/f).relative_to(ROOT)) for f in ['npu_sram_bb.v','dvp_capture.v','npu_core.v','bbox_overlay.v','person_detector_top.v'])+'\n')
print('Generated eight SRAM boundaries, production files unchanged:',DST)
