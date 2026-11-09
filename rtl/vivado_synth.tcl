# Run from repository root:
# FPGA_PART=<actual_part> vivado -mode batch -source rtl/vivado_synth.tcl
# Choose the actual part and desired clock before treating timing as meaningful.
if {![info exists ::env(FPGA_PART)] || $::env(FPGA_PART) eq ""} {
    error "Set FPGA_PART to the exact target FPGA part."
}
set script_dir [file dirname [file normalize [info script]]]
set report_dir [file join $script_dir build vivado]
file mkdir $report_dir
set target_top person_detector_top
if {[info exists ::env(RTL_TOP)]} { set target_top $::env(RTL_TOP) }
# Independent pixel27MHz and NPU100MHz clocks. Constraints do not generate clocks.
set pixel_period_ns [expr {1000.0/27.0}]
set npu_period_ns 10.0
if {[info exists ::env(PIXEL_PERIOD_NS)]} { set pixel_period_ns $::env(PIXEL_PERIOD_NS) }
if {[info exists ::env(NPU_PERIOD_NS)]} { set npu_period_ns $::env(NPU_PERIOD_NS) }
create_project -in_memory -part $::env(FPGA_PART)
foreach source_name {dvp_capture.v npu_core.v bbox_overlay.v person_detector_top.v} {
    read_verilog [file join $script_dir $source_name]
}
# Load the timing target before synthesis so synthesis can use it as well.
set clock_xdc [file join $report_dir clock.xdc]
set constraint_file [open $clock_xdc w]
puts $constraint_file [format {create_clock -name clk_pixel -period %.12f [get_ports clk_pixel]} $pixel_period_ns]
puts $constraint_file [format {create_clock -name clk_npu -period %.12f [get_ports clk_npu]} $npu_period_ns]
puts $constraint_file {set_clock_groups -name pixel_npu_async -asynchronous -group [get_clocks clk_pixel] -group [get_clocks clk_npu]}
close $constraint_file
read_xdc $clock_xdc
synth_design -top $target_top -part $::env(FPGA_PART)
source [file join $script_dir constraints dualclock_cdc.tcl]
report_utilization -hierarchical -file [file join $report_dir utilization.rpt]
report_ram_utilization -file [file join $report_dir ram_utilization.rpt]
report_timing_summary -file [file join $report_dir timing_synthesis.rpt]
report_drc -file [file join $report_dir drc.rpt]
report_clock_interaction -file [file join $report_dir clock_interaction.rpt]
report_cdc -details -file [file join $report_dir cdc.rpt]
write_checkpoint -force [file join $report_dir ${target_top}_synth.dcp]
# Explicitly list physical RAM primitives; confirm logical NPU arrays map here.
set ram_cells [get_cells -hierarchical -filter {REF_NAME =~ RAMB*}]
puts "Inferred BRAM primitive count: [llength $ram_cells]"
if {[llength $ram_cells] == 0} {
    puts "WARNING: No RAMB primitives inferred. Inspect memory inference warnings."
}
# Synthesis timing does not include routing and does not validate board I/O.
# Add actual pins/I/O delays and run implementation before hardware deployment.
