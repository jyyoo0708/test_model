# Source after synthesis. clock.xdc must define the two clock periods and:
# set_clock_groups -name pixel_npu_async -asynchronous \
#   -group [get_clocks clk_pixel] -group [get_clocks clk_npu]
# The Vivado driver reads this grouping BEFORE synth_design.
set cdc_first_stages [get_cells -hierarchical -filter {NAME =~ *_meta_q_reg}]
if {[llength $cdc_first_stages] < 7} {
    error "Expected seven CDC first stages; inspect hierarchy/name preservation."
}
# Cross-domain paths are excluded by asynchronous groups. No broad pin exception
# is needed: stage1->stage2 remains timed by its destination clock.
set_false_path -from [get_ports rst_n]
puts "CDC first stages: [llength $cdc_first_stages]; clocks asynchronous in clock.xdc"
# Input frame BRAM uses independent pixel-write/NPU-read ports. Check inference.
# Bbox banks are held until request/ack and only published at VSYNC. Async STA
# does not prove bundled-data settling; verify the physical transfer separately.
# Do not add a max-delay here and assume it overrides the asynchronous grouping.
