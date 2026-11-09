"""96x96 grayscale, stride-four detector with separate classification/regression paths."""
import copy
import torch
from torch import nn

class ConvBlock(nn.Sequential):
    def __init__(self,a,b,stride=1):
        super().__init__(nn.Conv2d(a,b,3,stride,1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True))

class AdvancedDetector(nn.Module):
    def __init__(self,width=1):
        super().__init__()
        a,b,c,d=[int(v*width) for v in (12,24,32,48)]
        self.c1=ConvBlock(1,a,2); self.c2=ConvBlock(a,b,2)
        self.c3=ConvBlock(b,c,2); self.c4=ConvBlock(c,d,2)
        self.context=ConvBlock(d,d)
        self.up1=nn.Upsample(scale_factor=2,mode='nearest')
        self.fuse1=ConvBlock(d+c,c)
        self.up2=nn.Upsample(scale_factor=2,mode='nearest')
        self.fuse2=ConvBlock(c+b,b); self.head=ConvBlock(b,b)
        self.regression=ConvBlock(b,b)
        self.heatmap=nn.Conv2d(b,1,1); self.offset=nn.Conv2d(b,2,1)
        self.size=nn.Conv2d(b,2,1); self.quality=nn.Conv2d(b,1,1)
        nn.init.constant_(self.heatmap.bias,-2.19)
        with torch.no_grad(): self.size.bias.copy_(torch.logit(torch.tensor([.18,.45])))

    def features(self,x):
        a=self.c1(x); b=self.c2(a); c=self.c3(b); d=self.context(self.c4(c))
        e=self.fuse1(torch.cat([self.up1(d),c],1))
        f=self.head(self.fuse2(torch.cat([self.up2(e),b],1)))
        r=self.regression(f)
        return torch.cat([self.heatmap(f),self.offset(r),self.size(r),self.quality(r)],1),f

    def forward(self,x): return self.features(x)[0]

    def fused(self):
        m=copy.deepcopy(self).eval()
        for block in m.modules():
            if isinstance(block,ConvBlock):
                block[0]=torch.nn.utils.fusion.fuse_conv_bn_eval(block[0],block[1]);block[1]=nn.Identity()
        return m


def memory_profile(model):
    """Explicit two-bank schedule, output-stationary convolution; no int32 SRAM partial sums."""
    rows=[]; hooks=[]
    def capture(name):
        def hook(m,ins,out):
            rows.append(dict(name=name,shape=list(out.shape),bytes_int8=out.numel(),
                weights=m.weight.numel(),biases=m.bias.numel(),
                macs=out.numel()*m.in_channels*m.kernel_size[0]*m.kernel_size[1]))
        return hook
    for n,m in model.named_modules():
        if isinstance(m,nn.Conv2d): hooks.append(m.register_forward_hook(capture(n)))
    with torch.inference_mode(): model(torch.zeros(1,1,96,96))
    for h in hooks:h.remove()
    w=sum(r['weights'] for r in rows); b=sum(r['biases'] for r in rows)
    # Student channel schedule only. Upsample reads directly to concat destinations.
    bank=32256
    schedule=[
        dict(stage='c1',A={'input':9216},B={'c1':27648}),
        dict(stage='c2',A={'c2_skip':13824},B={'c1':27648}),
        dict(stage='c3',A={'c2_skip':13824},B={'c3_skip':4608}),
        dict(stage='c4',A={'c2_skip':13824,'c4':1728},B={'c3_skip':4608}),
        dict(stage='context',A={'c2_skip':13824},B={'c3_skip':4608,'context':1728}),
        dict(stage='concat1',A={'c2_skip':13824,'concat1':11520},B={'c3_skip':4608,'context':1728}),
        dict(stage='fuse1',A={'c2_skip':13824,'concat1':11520},B={'fuse1':4608}),
        dict(stage='concat2',A={'c2_skip':13824,'fuse1_copy':4608},B={'concat2':32256}),
        dict(stage='fuse2',A={'fuse2':13824},B={'concat2':32256}),
        dict(stage='head',A={'fuse2':13824},B={'shared':13824}),
        dict(stage='regression',A={'regression':13824},B={'shared':13824}),
        dict(stage='outputs',A={'regression':13824,'packed_output':3456},B={'shared':13824,'head_scratch':1152})]
    peak=max(sum(s[k].values()) for s in schedule for k in ('A','B'))
    assert peak<=bank
    return dict(layers=rows,bank_bytes=bank,activation_bytes=2*bank,weights_int8=w,bias_int32=4*b,
                network_total_bytes=2*bank+w+4*b,limit_bytes=300000,proposal_limit_bytes=175640,
                macs=sum(r['macs'] for r in rows),schedule=schedule,
                assumptions=['Batch1 int8 activation/weight and int32 bias; float training/export.',
                'Output stationary PE registers hold int32 partial sums; no partial-sum SRAM.',
                'Explicit copies/free operations and direct upsample-to-concat writes; layer outputs reclaimed.',
                'Requantization coefficients, alignment and postprocessing storage not yet specified; excluded.'])
