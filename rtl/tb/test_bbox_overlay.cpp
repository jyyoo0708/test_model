#include "Vbbox_overlay.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>

static Vbbox_overlay d;
// Exact rational clocks: one scheduler unit = 1/27 ps.
// 100MHz halfperiod=135000 units; 27MHz halfperiod=500000 units.
static uint64_t now_units=0,next_npu=135000,next_pixel=12345;
static unsigned npu_ticks=0,pixel_ticks=0,box_count=0,generation=0,done_pulses=0;
static unsigned checked_pixels=0;
static void fail(const char* message) {
    std::fprintf(stderr,"FAIL post: npu=%u pixel=%u t=%llu: %s\n",npu_ticks,pixel_ticks,
                 (unsigned long long)now_units,message);
    std::exit(1);
}
static void require(bool condition,const char* message) {if(!condition)fail(message);}
static void observe_npu() {
    if(!d.rst_npu_n)return;
    if(d.o_boxes_done)++done_pulses;
    if(d.o_box_valid) {
        if(generation==1)fail("expected empty result");
        if(generation==2) {
            if(box_count>=50 || d.o_box_score!=(46399-box_count*100))fail("top50 score order");
            ++box_count;return;
        }
        if(generation==3) {
            // 64 isolated peaks: 63 equal scores and a final stronger peak.
            // Earliest 49 equal peaks survive, latest of the first50 is evicted.
            const unsigned ordinal=box_count==0?64:box_count;
            const int x=1+3*((ordinal-1)%8),y=1+3*((ordinal-1)/8);
            const int cx=x*262144+131072,cy=y*262144+131072;
            const int halfw=(21845*96)/2,halfh=(32768*96)/2;
            const int left=cx-halfw<0?0:cx-halfw,top=cy-halfh<0?0:cy-halfh;
            const int right=cx+halfw>6291456?6291456:cx+halfw;
            const int bottom=cy+halfh>6291456?6291456:cy+halfh;
            require(box_count<50,"tie result count overflow");
            require(d.o_box_score==(box_count==0?45999:44999),"equal-score output order");
            require(d.o_box_x1==(left>>16)&&d.o_box_y1==(top>>16)&&
                    d.o_box_x2==(((right>>8)+255)/256-1)&&
                    d.o_box_y2==(((bottom>>8)+255)/256-1),"raster ties or minimum tie eviction");
            ++box_count;return;
        }
        if(box_count==0 && !(d.o_box_x1==34&&d.o_box_y1==26&&d.o_box_x2==65&&d.o_box_y2==73))
            fail("first box coordinates");
        if(box_count==1 && !(d.o_box_x1==0&&d.o_box_y1==2&&d.o_box_x2==29&&d.o_box_y2==49))
            fail("second box coordinates");
        if(box_count>1)fail("overlap NMS");
        ++box_count;
    }
}
static void event_step() {
    now_units=next_npu<next_pixel?next_npu:next_pixel;
    bool npu_rise=false,pixel_rise=false;
    if(now_units==next_npu){d.clk_npu=!d.clk_npu;next_npu+=135000;npu_rise=d.clk_npu;}
    if(now_units==next_pixel){d.clk_pixel=!d.clk_pixel;next_pixel+=500000;pixel_rise=d.clk_pixel;}
    d.eval();
    if(npu_rise){++npu_ticks;observe_npu();}
    if(pixel_rise){
        ++pixel_ticks;
        if(d.rst_pixel_n)require(d.o_vsync==d.i_vsync&&d.o_hsync==d.i_hsync,"pixel sync alignment");
    }
    if(npu_ticks>2000000)fail("simulation timeout");
}
static void tick() {unsigned before=npu_ticks;while(npu_ticks==before)event_step();}
static void pixel_tick() {unsigned before=pixel_ticks;while(pixel_ticks==before)event_step();}
static bool synthetic_hit(int x,int y) {
    return ((x>=34)&&(x<=65)&&(y>=26)&&(y<=73)&&((x==34)||(x==65)||(y==26)||(y==73))) ||
           ((x>=0)&&(x<=29)&&(y>=2)&&(y<=49)&&((x==0)||(x==29)||(y==2)||(y==49)));
}
static void frame(bool boxes) {
    d.i_vsync=1;d.i_hsync=0;pixel_tick();
    for(int y=0;y<96;++y) {
        for(int x=0;x<96;++x) {
            d.i_hsync=1;pixel_tick();
            bool hit=boxes&&synthetic_hit(x,y);
            if(d.o_rgb!=(hit?0xff0000:0x123456)) {
                std::fprintf(stderr,"pixel %d,%d rgb=%06x expected=%06x\n",x,y,d.o_rgb,hit?0xff0000:0x123456);
                fail("frame publication or overlay pixel mismatch");
            }
            ++checked_pixels;
        }
        d.i_hsync=0;pixel_tick();
    }
    d.i_vsync=0;pixel_tick();
}
static void result() {
    unsigned done_before=done_pulses;
    for(int k=0;k<3456;++k) {
        d.i_result_valid=1;d.i_result_index=k;d.i_result_data=0;
        if(k==12*24+12)d.i_result_data=1;
        if(k==12*24+14)d.i_result_data=2;
        if(k==6*24+3)d.i_result_data=3;
        if(generation>=2 && k<576) {
            int x=k%24,y=k/24;
            d.i_result_data=(x%3==1 && y%3==1)?((y/3)*8+x/3+1):0;
        }
        // Exercise done coincident with final valid and LUT-pipeline drain.
        d.i_result_done=(k==3455);tick();
    }
    d.i_result_valid=0;d.i_result_done=0;
    while(done_pulses==done_before)tick();
}
static void config(unsigned address,unsigned value) {
    d.i_cfg_addr=address;d.i_cfg_wdata=value;d.i_cfg_we=1;tick();d.i_cfg_we=0;tick();
    require(d.o_cfg_rdata==value,"configuration write/readback");
}
static unsigned status() {d.i_cfg_addr=0x10;d.eval();return d.o_cfg_rdata;}
static void common_reset() {
    d.rst_npu_n=0;d.rst_pixel_n=0;
    d.i_result_valid=0;d.i_result_done=0;d.i_lut_we=0;d.i_cfg_we=0;
    d.i_vsync=0;d.i_hsync=0;
    for(int i=0;i<3;++i)pixel_tick();
    d.rst_npu_n=1;tick();tick();
    d.rst_pixel_n=1;
    for(int i=0;i<3;++i)pixel_tick();
}
int main(int argc,char** argv) {
    Verilated::commandArgs(argc,argv);
    if(argc>1)next_pixel=std::strtoull(argv[1],nullptr,10);
    const uint64_t initial_phase=next_pixel;
    d.i_rgb=0x123456;common_reset();
    require(!d.o_busy,"busy after reset");
    d.i_cfg_addr=0;d.eval();require(d.o_cfg_rdata==21299,"threshold reset value");
    for(int a=0;a<1536;++a) {
        d.i_lut_we=1;d.i_lut_addr=a;int c=a/256;
        d.i_lut_data=(c==1||c==2||c==4)?32768:((c==3)?21845:((c==5)?65535:0));
        if(c==0&&a%256==1)d.i_lut_data=60000;
        if(c==0&&a%256==2)d.i_lut_data=50000;
        if(c==0&&a%256==3)d.i_lut_data=45000;
        tick();
    }
    d.i_lut_we=0;frame(false);result();
    require(box_count==2,"expected two boxes after NMS");
    require(d.o_busy && (status()&2),"pending publication must remain busy");
    // Request has not passed its 2FF synchronizer at this first VSYNC.
    // It arrives during the frame; that entire frame must retain old data.
    frame(false);require(d.o_busy,"unpublished bank reused too early");
    frame(true);require(!d.o_busy,"acknowledged bank stays busy");
    require((status()>>2)==2,"active count not returned with ack");
    config(12,2);for(int i=0;i<3;++i)pixel_tick();frame(false);
    config(12,3);for(int i=0;i<3;++i)pixel_tick();frame(true);
    generation=1;config(0,65535);result();
    require(d.o_busy,"empty result must still publish a bank");
    frame(true);frame(false);
    require(!d.o_busy && status()==0,"empty bank publication/ack");
    generation=2;box_count=0;
    config(0,21299);config(8,0x0101);config(12,1);
    for(int a=0;a<256;++a) {
        d.i_lut_we=1;d.i_lut_addr=a;d.i_lut_data=(a>=1&&a<=64)?40000+a*100:0;tick();
    }
    d.i_lut_we=0;result();require(box_count==50,"best50 of64 isolated peaks");
    for(int i=0;i<50;++i)pixel_tick();
    require(d.o_busy && (status()&2),"pending bank cleared without VSYNC");
    // Aborting a pending transfer resets both ends; no phantom bank may publish.
    common_reset();require(!d.o_busy && status()==0,"common reset leaves phantom publication");
    frame(false);
    generation=3;box_count=0;
    config(0,21299);config(8,0x0101);config(12,1);
    for(int a=0;a<256;++a) {
        d.i_lut_we=1;d.i_lut_addr=a;d.i_lut_data=(a>=1&&a<=64)?(a==64?46000:45000):0;tick();
    }
    d.i_lut_we=0;result();require(box_count==50,"equal-score top50 count");
    common_reset();require(!d.o_busy,"reset after tie test");
    require(checked_pixels==73728,"RGB coverage accounting");
    std::printf("PASS bbox_overlay dual-clock27/100MHz phase=%llu: NMS, best50 and raster ties, config CDC, empty/reset, atomic publication/ack, %u exact RGB pixels; npu=%u pixel=%u\n",
                (unsigned long long)initial_phase,checked_pixels,npu_ticks,pixel_ticks);
    d.final();return 0;
}
