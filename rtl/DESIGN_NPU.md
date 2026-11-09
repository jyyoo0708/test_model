# NPU contract and architecture — pixel27MHz / NPU100MHz

The implementation uses plain synthesizable Verilog, two independent clocks,
signed INT8 activations/weights and INT32 biases. The complete trained student
graph remains c1,c2,c3,c4,context,fuse1,fuse2,head,regression,heatmap,offset,size,
quality. Quantized weights, descriptors, scales and output ordering are unchanged.

## Interface and ownership

- `clk_pixel` is the27MHz pixel clock. `i_pixel_we`, `i_pixel_addr[13:0]` and
  signed `i_pixel_data[7:0]` are synchronous to this clock.
- `clk_npu` is the100MHz compute clock. `i_frame_start`, all model preload
  ports, `o_busy`, result stream and `o_done` are synchronous to this clock.
- `rst_n` asynchronously resets NPU control and must be synchronously released
  to `clk_npu` by the parent reset synchronizer. The parent resets both domains
  together. Neither SRAM array nor a RAM output register is reset.
- The top owns the frame protocol: complete pixel writes, transfer a request
  toggle through two NPU synchronizer flops, then keep the entire input frame
  unchanged until the result/postprocessing completion returns to capture.
  Capture cannot overwrite the owned frame. No per-byte synchronizer is used.
- A raw asynchronous `i_frame_start` must never be connected directly to the
  core. The core accepts an already synchronized one-NPU-clock pulse.
- Model preload writes and overlapping starts while `o_busy` are ignored.
  Pixel writes are **not** gated by an asynchronous NPU busy signal. Writing an
  owned input frame is a protocol violation and must be prevented by the parent.
- Reset aborts both input copying and computation. SRAM contents survive; a
  restarted operation recopies all9216 input bytes before using them.

## Clock-domain memory boundary

A dedicated9216x8 `input_frame_mem` is the only dual-clock memory in the core.
Its write-only port runs on `clk_pixel`; its registered read port runs on
`clk_npu` with exactly one clock of read latency. The frame request/ownership
handshake guarantees no same-address cross-clock read/write collision during
use. Pixel data is quantized as `(pixel*127+127)/255`, scale1/127, zero point0.

The NPU start sequence is:

1. `p_idle` accepts `i_frame_start`, initializes copy index0.
2. `p_input_request` requests input-frame SRAM[index].
3. `p_input_use` writes that byte to activation SRAM[index], then advances.
4. After index9215, descriptor fetch and the original13-layer controller run.

Exactly18432 NPU clocks are added for this copy. Existing activation memory
remains wholly in the100MHz domain and has one muxed synchronous read port
and one muxed synchronous write port. There are no different-clock writers
to the activation array. The source frame stays frozen for the whole operation,
not merely for the copy interval, to simplify parent ownership.

## Quantization, graph and descriptor contract

Batch normalization is folded before quantization. Each output channel uses a
symmetric weight scale. Bias is rounded by input_scale*weight_scale to INT32.
Each convolution output is independently requantized; the two inputs of each
concatenation share a calibrated scale.

The preload bus selects space0 INT8 weights, space1 INT32 bias, space2 32-bit
descriptors, space3 signed32-bit requantization multipliers. Address is an
element index. Each channel's bias and multiplier share the same address.
No design-side file initialization is present; preload must finish before start.

Each convolution descriptor has16 words: first-input base, second-input base,
output base, weight base, parameter base, logical input width, logical input
height, first-input channels, second-input channels, output width, output height,
output channels, kernel width(1 or3), stride(1 or2), first-source upsample enable,
ReLU enable. Source1 can be nearest-neighbor upsampled by2; source2 is the
same-resolution skip tensor. Padding is implicit. Descriptors do not change
for the added frame-copy RAM because convolution still reads activation base0.

One output-stationary signed multiplier and INT32 accumulator execute one
product per two clocks. Output is `clip_int8((acc*multiplier+2^23) >>>24)`
followed by optional ReLU. Ties round toward positive infinity. The positive
multiplier is round(input_scale*weight_scale/output_scale*2^24). There is no
partial-sum SRAM. All graph tensors are retained; virtual concatenation avoids
copies. Diagnostic float scales in manifest.json are not hardware float values.

`o_result_valid` accompanies index0..3455, six24x24 planes in channel-major
order. `o_done` pulses on the NPU clock after the last result. The parent
postprocessor is on the same NPU clock; image rendering is on the pixel clock.

## BRAM replacement and capacity

| Storage | Elements | Width | Bytes | Clock |
|---|---:|---:|---:|---|
| Input frame CDC RAM |9216|8|9216|pixel write / NPU read|
| Activation, original input copy and output |108288|8|108288|NPU|
| Weight |89820|8|89820|NPU|
| Bias |274|32|1096|NPU|
| Requantization multiplier |274|32|1096|NPU|
| Layer descriptor |208|32|832|NPU|
| **Total NPU SRAM** | | |**210348**| |
| Post LUT/result SRAM (separate block) | | |9984|NPU|
| **Total SRAM** | | |**220332**| |

Small candidate/bbox and control registers are extra; logical storage remains
below300KB. Physical BRAM blocks round capacities according to width/depth.
Every RAM read requires latency1. Disable any additional Block Memory Generator
output register, or add matching controller wait states and rerun verification.
The frame RAM needs independent clocks; other NPU RAMs use a common NPU clock.
Do not reset arrays or replace the dedicated frame RAM with an asynchronous
combinational read bus.

For signedINT8 weights in[-127,127], activations in[-128,127], the conservative
channel accumulator bound is abs(bias_q)+K*K*Cin*128*127. The maximum fan-in720
for fuse1 bounds the sum of products by11,704,320 before bias. The unchanged
`generated/quantization_report.json` confirms actual bias-inclusive INT32 and
INT64 requantization bounds for the fixed model.

## Latency and validation

Original graph start-to-done:40,947,105 NPU clocks. The dual-clock implementation
adds18432 copy clocks, giving **40,965,537 NPU clocks**, about409.65537ms at100MHz
and a core processing upper bound of about2.4411 inferences/s. Capture at27MHz,
postprocessing, frame publication, padding reinitialization and next-frame wait
further reduce system inference FPS.100MHz is a timing target until actual FPGA
implementation meets timing; simulation establishes cycle behavior only.

Unit verification uses exact rational asynchronous clocks: one simulator time
unit is1/27ns, half-periods500(pixel)/135(NPU), and pixel phase17units prevents
coincident edges. Pixel writes occur only on pixel edges; preload/start/debug
checks occur only on NPU edges. The bench checks all3456 output bytes against
the independent NumPy INT64 convolution reference. It aborts once during frame
copy and once after copy/compute begins, verifies reset recovery, and attempts
busy model writes plus an overlapping start. It never illegally modifies an
owned pixel frame.

```bash
verilator --lint-only --language 1364-2005 -Wall -Wno-UNUSED --top-module npu_core rtl/npu_core.v
verilator --cc --exe --build --language 1364-2005 -Wall -Wno-UNUSED --top-module npu_core --Mdir rtl/build/dualclock_npu rtl/npu_core.v "$PWD/rtl/tb/tb_npu.cpp" -CFLAGS '-std=c++11 -O3' -o test
rtl/build/dualclock_npu/test
```

A full40-million-clock waveform is omitted by this unit bench. The integrated
pipeline bench provides bounded image-overlay waveforms and validates the
parent CDC request/completion paths. No metastability analog model or physical
CDC/STA closure is implied by RTL simulation.

Executed result: **PASS3456/3456 exact bytes,40,965,537 NPU clocks**, asynchronous
27/100MHz clocks, copy/compute reset abort and busy model-write/start rejection.
Log: `rtl/build/dualclock_npu/simulation.log`. Strict Verilog lint completed
without findings after unused-bit notices were excluded (`-Wno-UNUSED`).
