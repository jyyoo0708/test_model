#include "Vdvp_capture.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
#ifndef SENSOR_W
#define SENSOR_W 96
#define SENSOR_H 96
#define INPUT_TYPE 0
#endif
static Vdvp_capture d;
static std::vector<int> memory(9216, -1);
static unsigned starts=0, errors=0, writes=0;
static void require(bool ok,const char* msg) { if(!ok) throw std::runtime_error(msg); }
static void tick() {
    d.clk=0;d.eval();d.clk=1;d.eval();
    if(d.rst_n && d.o_pixel_we) {
        require(d.o_pixel_addr<9216,"out-of-range write");
        memory[d.o_pixel_addr]=d.o_pixel_data; ++writes;
    }
    if(d.o_frame_start) ++starts;
    if(d.o_frame_error) ++errors;
}
static uint32_t pixel(int x,int y) {
    unsigned a=(x*7+y*11)%256;
    if(INPUT_TYPE==0) return a;
    if(INPUT_TYPE==1) return (a<<16)|(((a+33)%256)<<8)|((a+79)%256);
    return ((a&31)<<11)|(((a+9)&63)<<5)|((a+17)&31);
}
static int gray(uint32_t v) {
    if(INPUT_TYPE==0) return v&255;
    int r,g,b;
    if(INPUT_TYPE==1) {r=(v>>16)&255;g=(v>>8)&255;b=v&255;}
    else {r=((v>>11)&31)*8+((v>>13)&7);g=((v>>5)&63)*4+((v>>9)&3);b=(v&31)*8+((v>>2)&7);}
    return (77*r+150*g+29*b+128)>>8;
}
static void wait_ready() {for(int n=0;n<9220 && !d.o_ready;n++)tick();require(d.o_ready,"ready timeout");}
static void frame(int rows) {
    d.i_vsync=1;d.i_hsync=0;tick();
    for(int y=0;y<rows;y++) {
        for(int x=0;x<SENSOR_W;x++) {d.i_hsync=1;d.i_data=pixel(x,y);tick();}
        d.i_hsync=0;tick();
    }
    tick();d.i_vsync=0;tick();tick();
}
int main(int argc,char** argv) {
    Verilated::commandArgs(argc,argv);
    try {
        d.rst_n=0;tick();d.rst_n=1;wait_ready();
        require(writes==9216,"padding writes mismatch");
        frame(SENSOR_H);
        require(starts==1 && errors==0,"nominal frame rejected");
        int mx=SENSOR_W>SENSOR_H?SENSOR_W:SENSOR_H;
        int w=(SENSOR_W*96+mx/2)/mx,h=(SENSOR_H*96+mx/2)/mx;
        int left=(96-w)/2,top=(96-h)/2;
        for(int y=0;y<96;y++) for(int x=0;x<96;x++) {
            int expected=57;
            if(x>=left && x<left+w && y>=top && y<top+h) {
                int sx=(x-left)*SENSOR_W/w,sy=(y-top)*SENSOR_H/h;
                expected=(gray(pixel(sx,sy))*127+127)/255;
            }
            require(memory[y*96+x]==expected,"sample/letterbox mismatch");
        }
        unsigned saved=writes;frame(SENSOR_H);
        require(starts==1 && writes==saved,"busy frame not skipped");
        d.i_inference_done=1;tick();d.i_inference_done=0;wait_ready();
        frame(SENSOR_H-1);require(starts==1 && errors==1,"truncated frame accepted");
        wait_ready();frame(SENSOR_H);require(starts==2,"recovery failed");
        std::printf("PASS capture %dx%d type%d: pixels, padding, busy-drop, truncated frame, recovery\n",SENSOR_W,SENSOR_H,INPUT_TYPE);
        d.final();return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
