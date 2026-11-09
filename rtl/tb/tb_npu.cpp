#include "Vnpu_core.h"
#include "verilated.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint32_t> readhex(const std::string &path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Missing " + path);
    std::vector<uint32_t> r;
    std::string s;
    while (f >> s) r.push_back(static_cast<uint32_t>(std::stoul(s,nullptr,16)));
    return r;
}

// Exact rational schedule: one time unit=1/27 ns. Half periods are135units
// (100MHz NPU) and500units(27MHz pixel). Pixel phase17 prevents coincident edges.
class Clocks {
    Vnpu_core &d;
    uint64_t next_npu=135, next_pixel=17;
public:
    uint64_t npu_cycles=0, pixel_cycles=0, time_units=0;
    explicit Clocks(Vnpu_core &model):d(model) {
        d.clk_npu=0;d.clk_pixel=0;d.eval();
    }
    unsigned step() {
        unsigned risen=0;
        if (next_npu<next_pixel) {
            time_units=next_npu;next_npu+=135;
            d.clk_npu=!d.clk_npu;
            if(d.clk_npu){++npu_cycles;risen|=1;}
        } else {
            time_units=next_pixel;next_pixel+=500;
            d.clk_pixel=!d.clk_pixel;
            if(d.clk_pixel){++pixel_cycles;risen|=2;}
        }
        d.eval();return risen;
    }
    void tick_npu(){while(!(step()&1)) {}}
    void tick_pixel(){while(!(step()&2)) {}}
};

int main(int argc,char **argv) {
    Verilated::commandArgs(argc,argv);
    Vnpu_core d;
    d.rst_n=0;d.i_frame_start=0;d.i_pixel_we=0;d.i_load_we=0;
    Clocks clocks(d);
    auto tick=[&]() {clocks.tick_npu();};
    tick();tick();d.rst_n=1;tick();
    const char *names[]={"weights","bias","descriptors","requant"};
    for(unsigned space=0;space<4;++space) {
        auto data=readhex(std::string("rtl/generated/")+names[space]+".hex");
        d.i_load_space=space;d.i_load_we=1;
        for(unsigned a=0;a<data.size();++a) {d.i_load_addr=a;d.i_load_data=data[a];tick();}
    }
    d.i_load_we=0;
    auto input=readhex("rtl/generated/input_q.hex");
    auto golden=readhex("rtl/generated/golden_output.hex");
    if(input.size()!=9216||golden.size()!=3456)throw std::runtime_error("Reference dimensions");
    d.i_pixel_we=1;
    for(unsigned a=0;a<input.size();++a) {
        d.i_pixel_addr=a;d.i_pixel_data=input[a];clocks.tick_pixel();
    }
    d.i_pixel_we=0;
    // Input ownership is held by the parent frame handshake for the whole run.
    // Exercise reset once during copy and once after copy/inference has begun.
    for(unsigned abort_after: {20u,19000u}) {
        d.i_frame_start=1;tick();d.i_frame_start=0;
        for(unsigned i=0;i<abort_after;++i)tick();
        d.rst_n=0;tick();d.rst_n=1;tick();
        if(d.o_busy||d.o_result_valid||d.o_done)throw std::runtime_error("Reset failed");
    }
    d.i_frame_start=1;tick();d.i_frame_start=0;
    const uint64_t start=clocks.npu_cycles;
    unsigned outputs=0;
    std::ofstream dump("rtl/build/npu_output.hex");
    while(!d.o_done) {
        // NPU-domain model writes and overlapping starts are rejected while busy.
        // Pixel writes are deliberately inactive: the source owns a stable frame.
        d.i_load_we=clocks.npu_cycles-start<100;
        d.i_load_space=0;d.i_load_addr=0;d.i_load_data=0;
        d.i_frame_start=clocks.npu_cycles-start==10;
        tick();
        if(d.o_result_valid) {
            if(outputs>=golden.size()||d.o_result_index!=outputs || d.o_result_data!=golden.at(outputs)) {
                std::cerr<<"Mismatch index "<<outputs<<" RTL "<<int(d.o_result_data)
                         <<" golden "<<(outputs<golden.size()?golden.at(outputs):999)<<"\n";
                return 1;
            }
            dump<<std::hex<<int(d.o_result_data)<<"\n";
            ++outputs;
        }
        if(clocks.npu_cycles-start>50000000)throw std::runtime_error("NPU watchdog");
    }
    if(outputs!=3456)throw std::runtime_error("Missing output samples");
    if(clocks.npu_cycles-start!=40965537)throw std::runtime_error("Unexpected inference cycles");
    std::cout<<"PASS: "<<outputs<<" bit-exact outputs, "<<std::dec<<clocks.npu_cycles-start
             <<" NPU cycles, asynchronous27/100MHz clocks, copy/compute reset abort, busy model-write/start rejection\n";
    d.final();return 0;
}
