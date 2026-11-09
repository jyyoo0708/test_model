#include "Vperson_detector_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
static VerilatedVcdC* waveform=nullptr;
static Vperson_detector_top* dut=nullptr;
// One unit = 1/27 ps. Exact unrelated27/100MHz clock edges, nonzero phase.
static uint64_t now_units=0,next_npu=135000,next_pixel=517321;
static const uint64_t npu_half=135000,pixel_half=500000;
static uint64_t npu_cycles=0,pixel_cycles=0;
static bool tracing=false,trace_window=false,done=false;
static unsigned outputs=0,boxcount=0,npu_done_count=0,boxes_done_count=0;
static std::vector<uint32_t> golden;
static std::ofstream boxfile;
static void require(bool ok,const std::string& msg){if(!ok)throw std::runtime_error(msg);}
static std::vector<uint32_t> hexfile(const char* name){
    std::ifstream f(std::string("rtl/generated/")+name);require(bool(f),std::string("missing ")+name);
    std::vector<uint32_t> a;std::string s;while(f>>s)a.push_back(uint32_t(std::stoull(s,nullptr,16)));return a;
}
static unsigned event(){
    now_units=next_npu<next_pixel?next_npu:next_pixel;
    bool edge_npu=(now_units==next_npu),edge_pixel=(now_units==next_pixel);
    bool rise_npu=edge_npu&&!dut->clk_npu,rise_pixel=edge_pixel&&!dut->clk_pixel;
    if(edge_npu){dut->clk_npu=!dut->clk_npu;next_npu+=npu_half;}
    if(edge_pixel){dut->clk_pixel=!dut->clk_pixel;next_pixel+=pixel_half;}
    dut->eval();
    if(tracing&&trace_window)waveform->dump(now_units);
    if(rise_npu){
        ++npu_cycles;
        if(dut->rst_n){
            if(dut->o_result_valid){
                require(outputs<golden.size(),"too many NPU outputs");
                require(dut->o_result_index==outputs,"NPU output ordering");
                require(dut->o_result_data==golden[outputs],"NPU mismatch index="+std::to_string(outputs));++outputs;
            }
            if(dut->o_npu_done)++npu_done_count;
            if(dut->o_box_valid){
                boxfile<<unsigned(dut->o_box_x1)<<' '<<unsigned(dut->o_box_y1)<<' '<<unsigned(dut->o_box_x2)<<' '<<unsigned(dut->o_box_y2)<<' '<<unsigned(dut->o_box_score)<<'\n';++boxcount;
            }
            if(dut->o_boxes_done){done=true;++boxes_done_count;}
        }
    }
    if(rise_pixel){
        ++pixel_cycles;
        if(dut->rst_n){
            require(!dut->o_frame_error,"unexpected capture error");
            require(dut->o_vsync==dut->i_vsync&&dut->o_hsync==dut->i_hsync,"pixel sync mismatch");
        }
    }
    return (rise_npu?1u:0u)|(rise_pixel?2u:0u);
}
static void npu_tick(bool changed=false){if(changed)dut->eval();while(!(event()&1u)){} }
static void pixel_tick(bool changed=false){if(changed)dut->eval();while(!(event()&2u)){} }
static void load(unsigned space,const char* name){
    auto a=hexfile(name);dut->i_load_we=1;dut->i_load_space=space;
    for(unsigned i=0;i<a.size();++i){dut->i_load_addr=i;dut->i_load_data=a[i];npu_tick(true);}
    dut->i_load_we=0;npu_tick(true);
}
static void config(unsigned addr,unsigned data){
    dut->i_cfg_addr=addr;dut->i_cfg_wdata=data;dut->i_cfg_we=1;npu_tick(true);dut->i_cfg_we=0;npu_tick(true);
    require(dut->o_cfg_rdata==data,"config readback mismatch");
}
static void wait_ready(){for(int n=0;n<10000&&!dut->o_ready;++n)pixel_tick();require(dut->o_ready,"capture ready timeout");}
static void frame(const std::vector<uint32_t>& pixels,bool render,bool must_be_busy=false){
    std::ofstream ppm;
    if(render){ppm.open("rtl/results/rtl_overlay.ppm",std::ios::binary);ppm<<"P6\n96 96\n255\n";}
    dut->i_vsync=1;dut->i_hsync=0;pixel_tick(true);
    for(int y=0;y<96;++y){
        for(int x=0;x<96;++x){
            dut->i_hsync=1;dut->i_data=pixels[y*96+x];pixel_tick(true);
            if(must_be_busy)require(!dut->o_ready,"busy frame was accepted");
            if(render){char rgb[3]={char(dut->o_rgb>>16),char(dut->o_rgb>>8),char(dut->o_rgb)};ppm.write(rgb,3);}
            else {unsigned v=pixels[y*96+x];require(dut->o_rgb==v*0x010101u,"bypass mismatch");}
        }
        dut->i_hsync=0;pixel_tick(true);
    }
    pixel_tick(true);dut->i_vsync=0;pixel_tick(true);pixel_tick(true);
}
static void check_input(const std::vector<uint32_t>& pixels,bool copied){
    for(unsigned i=0;i<pixels.size();++i){
        unsigned q=(pixels[i]*127+127)/255;
        unsigned got=copied?dut->person_detector_top__DOT__u_npu__DOT__activation_mem[i]:dut->person_detector_top__DOT__u_npu__DOT__input_frame_mem[i];
        require(got==q,std::string(copied?"activation copy":"frame SRAM")+" mismatch index="+std::to_string(i)+" got="+std::to_string(got)+" expected="+std::to_string(q));
    }
}
static void check_parameters(){
    auto w=hexfile("weights.hex"),b=hexfile("bias.hex"),m=hexfile("requant.hex"),d=hexfile("descriptors.hex");
    for(unsigned i=0;i<w.size();++i)require(dut->person_detector_top__DOT__u_npu__DOT__weight_mem[i]==w[i],"weight preload mismatch "+std::to_string(i));
    for(unsigned i=0;i<b.size();++i)require(dut->person_detector_top__DOT__u_npu__DOT__bias_mem[i]==b[i],"bias preload mismatch "+std::to_string(i));
    for(unsigned i=0;i<m.size();++i)require(dut->person_detector_top__DOT__u_npu__DOT__multiplier_mem[i]==m[i],"requant preload mismatch "+std::to_string(i));
    for(unsigned i=0;i<d.size();++i)require(dut->person_detector_top__DOT__u_npu__DOT__descriptor_mem[i]==d[i],"descriptor preload mismatch "+std::to_string(i));
}
static void reset(){
    dut->rst_n=0;dut->i_vsync=0;dut->i_hsync=0;dut->i_load_we=0;dut->i_cfg_we=0;dut->eval();
    for(int i=0;i<4;++i)pixel_tick();dut->rst_n=1;dut->eval();
    for(int i=0;i<5;++i)pixel_tick();
}
int main(int argc,char** argv){
    Verilated::commandArgs(argc,argv);Verilated::traceEverOn(true);
    dut=new Vperson_detector_top;waveform=new VerilatedVcdC;
    for(int i=1;i<argc;++i){
        if(std::string(argv[i])=="--trace")tracing=true;
        if(std::string(argv[i])=="--phase"&&i+1<argc)next_pixel=500000+std::stoull(argv[++i]);
    }
    if(tracing){dut->trace(waveform,4);waveform->open("rtl/build/tb_pipeline.vcd");}
    try{
        boxfile.open("rtl/results/rtl_boxes.txt");golden=hexfile("golden_output.hex");require(golden.size()==3456,"bad reference output");
        auto pixels=hexfile("input_pixels.hex");require(pixels.size()==9216,"bad input image");
        reset();load(0,"weights.hex");load(1,"bias.hex");load(2,"descriptors.hex");load(3,"requant.hex");load(4,"post_lut.hex");check_parameters();
        wait_ready();frame(pixels,false);for(int i=0;i<1000;++i)npu_tick();
        require(outputs==0,"unexpected early output");reset();wait_ready();check_parameters();
        config(0,21299);config(4,32768);config(8,0x1808);config(12,3);
        frame(pixels,false);check_input(pixels,false);uint64_t begin=npu_cycles,begin_time=now_units;
        // Distinct busy-frame pixels stress ownership during input RAM copy.
        auto inverse=pixels;for(auto& v:inverse)v=255-v;frame(inverse,false,true);check_input(pixels,false);check_input(pixels,true);
        while(!done&&npu_cycles-begin<50000000)npu_tick();
        require(done,"inference/post timeout");require(outputs==3456,"incomplete output");
        require(npu_done_count==1&&boxes_done_count==1,"duplicate/missing done event across CDC");
        uint64_t latency=npu_cycles-begin,latency_units=now_units-begin_time;
        wait_ready();trace_window=true;frame(pixels,true);trace_window=false;
        config(12,2);for(int i=0;i<4;++i)pixel_tick();frame(pixels,false);
        require(npu_done_count==1&&boxes_done_count==1,"stale CDC event retriggered");
        boxfile.close();dut->final();if(tracing)waveform->close();
        std::printf("PASS dual-clock pipeline: 3456 NPU bytes, %u boxes, reset recovery, busy ownership,27/100MHz CDC, sync/bypass/config; %llu inference+post NPU cycles\n",boxcount,(unsigned long long)latency);
        std::ofstream stats("rtl/results/cycles.json");
        stats<<"{\"clock_pixel_hz\":27000000,\"clock_npu_hz\":100000000,\"inference_post_cycles\":"<<latency
             <<",\"inference_post_seconds\":"<<double(latency_units)/27000000000000.0
             <<",\"npu_core_cycles\":40965537,\"input_copy_cycles\":18432,\"total_npu_cycles\":"<<npu_cycles
             <<",\"total_pixel_cycles\":"<<pixel_cycles<<",\"reset_recovery_checked\":true,\"busy_frame_drop_checked\":true}\n";
        delete dut;delete waveform;return 0;
    }catch(const std::exception& e){
        std::fprintf(stderr,"FAIL %s at NPU cycle=%llu pixel=%llu output=%u\n",e.what(),(unsigned long long)npu_cycles,(unsigned long long)pixel_cycles,outputs);
        if(tracing)waveform->close();delete dut;delete waveform;return 1;
    }
}
