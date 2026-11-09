if {[catch {
set run_root [file normalize $env(NPU_DC_ROOT)]
set out_dir [file normalize $env(NPU_DC_OUT)]
set rtl_dir [file join $out_dir rtl]
set topname person_detector_top
set stdcell /tools/lib/ln28lpp/primitive/ln28lpp_sc_9t_base_lvt_c130_FE-Common_sec190802_0202/LIBERTY/sc9_cmos28lpp_base_lvt_c34_ss_nominal_max_0p950v_m25c_sadhm.db
file mkdir $out_dir
define_design_lib WORK -path [file join $out_dir work]
set search_path [concat [list $rtl_dir /tools/synopsys/syn/I-2013.12-SP4/libraries/syn] $search_path]
set target_library [list $stdcell]
set synthetic_library [list dw_foundation.sldb]
set link_library [concat [list * dw_foundation.sldb] $target_library]
set_svf [file join $out_dir ${topname}.svf]
catch {set_host_options -max_cores 4}
set sources [list]
foreach name {npu_sram_bb.v dvp_capture.v npu_core.v bbox_overlay.v person_detector_top.v} {
    lappend sources [file join $rtl_dir $name]
}
puts "BB_STAGE: analyze"
if {![analyze -format verilog $sources]} {puts "BB_FAILED: analyze"; exit 2}
puts "BB_STAGE: elaborate"
if {![elaborate $topname]} {puts "BB_FAILED: elaborate"; exit 3}
current_design $topname
puts "BB_STAGE: link"
if {![link]} {puts "BB_FAILED: link"; exit 4}

set memories [get_cells -hierarchical -filter {ref_name =~ npu_sram_bb*}]
if {[sizeof_collection $memories] != 8} {error "Expected eight SRAM black boxes"}
set_dont_touch $memories
set mf [open [file join $out_dir memories.rpt] w]
puts $mf "SRAM black boxes: eight; total excluded storage220332bytes. No macro area/timing model."
foreach_in_collection cell $memories {
    set ref [get_attribute $cell ref_name]
    set_dont_touch [get_designs $ref]
    puts $mf "[get_object_name $cell] reference=$ref is_black_box=[get_attribute $cell is_black_box]"
}
close $mf
redirect [file join $out_dir check_design_pre.rpt] {check_design}
redirect [file join $out_dir resources_pre.rpt] {report_resources -hierarchy}
redirect [file join $out_dir references_pre.rpt] {report_reference -hierarchy}

create_clock -name clk_pixel -period [expr {1000.0/27.0}] [get_ports clk_pixel]
create_clock -name clk_npu -period 10.0 [get_ports clk_npu]
set_clock_groups -name pixel_npu_async -asynchronous -group [get_clocks clk_pixel] -group [get_clocks clk_npu]
set_clock_uncertainty -setup 0.1 [all_clocks]
set_clock_uncertainty -hold 0.05 [all_clocks]
set_clock_transition 0.1 [all_clocks]
# Fix actual short-path hold violations by inserting mapped delay cells.
set_fix_hold [all_clocks]
set pixel_inputs [get_ports {i_vsync i_hsync i_data*}]
set npu_inputs [get_ports {i_load_* i_cfg_*}]
set pixel_outputs [get_ports {o_ready o_busy o_frame_error o_vsync o_hsync o_rgb*}]
set npu_outputs [remove_from_collection [all_outputs] $pixel_outputs]
set_input_delay -max 1.0 -clock clk_pixel $pixel_inputs
set_input_delay -min 0.0 -clock clk_pixel $pixel_inputs
set_input_delay -max 1.0 -clock clk_npu $npu_inputs
set_input_delay -min 0.0 -clock clk_npu $npu_inputs
set_input_transition 0.2 [add_to_collection $pixel_inputs $npu_inputs]
set_output_delay -max 1.0 -clock clk_pixel $pixel_outputs
set_output_delay -min 0.0 -clock clk_pixel $pixel_outputs
set_output_delay -max 1.0 -clock clk_npu $npu_outputs
set_output_delay -min 0.0 -clock clk_npu $npu_outputs
set_load 0.01 [all_outputs]
set_false_path -from [get_ports rst_n]
set_max_fanout 20 [current_design]

# Clock groups exclude all inter-domain setup/hold paths. First-stage inventory
# is retained for structural review; stage1->stage2 remains timed in each domain.
set first_stages [get_cells -hierarchical -filter {full_name =~ *_meta_q_reg}]
if {[sizeof_collection $first_stages] < 7} {error "Missing CDC synchronizers"}
redirect [file join $out_dir clock_groups_pre.rpt] {report_clock -groups}

# SRAMs have NO timing arcs. Add explicit external-logic budgets so addresses,
# write data and read-data logic are not silently absent from timing reports.
# 9ns NPU /36ns pixel are assumed budgets reserving ~1ns for unknown macro overhead.
set memory_outputs [get_pins -of_objects $memories -filter {pin_direction == out}]
set memory_inputs [get_pins -of_objects $memories -filter {pin_direction == in && full_name !~ */clk_*}]
set frame_memory [get_cells u_npu/u_mem_input_frame]
set frame_write_inputs [get_pins -of_objects $frame_memory -filter {pin_direction == in && full_name =~ */i_write*}]
set npu_memory_inputs [remove_from_collection $memory_inputs $frame_write_inputs]
if {[sizeof_collection $memory_outputs] == 0 || [sizeof_collection $npu_memory_inputs] == 0 || [sizeof_collection $frame_write_inputs] == 0} {error "Missing SRAM boundary timing endpoints"}
set_max_delay 9.0 -from $memory_outputs -group_path SRAM_READ_NPU
set_min_delay 0.0 -from $memory_outputs
set_max_delay 9.0 -to $npu_memory_inputs -group_path SRAM_WRITE_NPU
set_min_delay 0.0 -to $npu_memory_inputs
set_max_delay 36.0 -to $frame_write_inputs -group_path SRAM_WRITE_PIXEL
set_min_delay 0.0 -to $frame_write_inputs

redirect [file join $out_dir check_timing_pre.rpt] {check_timing}
redirect [file join $out_dir clocks.rpt] {report_clock}
write_sdc [file join $out_dir ${topname}_pre.sdc]
puts "BB_STAGE: compile"
if {![compile]} {puts "BB_FAILED: compile"; exit 6}
puts "BB_STAGE: reports"
change_names -rules verilog -hierarchy
set memories_post [get_cells -hierarchical -filter {ref_name =~ npu_sram_bb*}]
if {[sizeof_collection $memories_post] != 8} {error "A SRAM boundary was optimized away"}
set verilogout_no_tri true
write -format verilog -hierarchy -output [file join $out_dir ${topname}_syn.v]
write -format ddc -hierarchy -output [file join $out_dir ${topname}.ddc]
write_sdc [file join $out_dir ${topname}.sdc]
redirect [file join $out_dir area.rpt] {report_area -hierarchy}
redirect [file join $out_dir timing_setup.rpt] {report_timing -delay_type max -max_paths 30 -nworst 1}
redirect [file join $out_dir timing_hold.rpt] {report_timing -delay_type min -max_paths 20 -nworst 1}
foreach group {clk_npu clk_pixel SRAM_READ_NPU SRAM_WRITE_NPU SRAM_WRITE_PIXEL} {
    redirect [file join $out_dir timing_${group}.rpt] {report_timing -group $group -delay_type max -max_paths 10 -nworst 1}
}
set npu_registers [filter_collection [all_registers] {full_name =~ u_npu/*}]
redirect [file join $out_dir timing_npu_core_internal.rpt] {report_timing -group clk_npu -to $npu_registers -delay_type max -max_paths 10 -nworst 1}
redirect [file join $out_dir constraints.rpt] {report_constraint -all_violators}
redirect [file join $out_dir qor.rpt] {report_qor}
redirect [file join $out_dir references.rpt] {report_reference -hierarchy}
redirect [file join $out_dir design.rpt] {report_design}
redirect [file join $out_dir check_design_post.rpt] {check_design}
redirect [file join $out_dir check_timing_post.rpt] {check_timing}
# Save all normal async-constrained artifacts BEFORE the separate payload audit.
redirect [file join $out_dir clock_groups.rpt] {report_clock -groups}
redirect [file join $out_dir timing_async_npu_to_pixel.rpt] {report_timing -from [get_clocks clk_npu] -to [get_clocks clk_pixel] -delay_type max -max_paths 10}
redirect [file join $out_dir timing_async_pixel_to_npu.rpt] {report_timing -from [get_clocks clk_pixel] -to [get_clocks clk_npu] -delay_type max -max_paths 10}
# Diagnostic only: remove group IN MEMORY for explicit held-data settling check.
# This view is not compiled or written as the implementation DDC/SDC/netlist.
set payload [get_cells -hierarchical -filter {full_name =~ *u_post*box_sensor_*_q_reg* || full_name =~ *u_post*publish_bank_q_reg* || full_name =~ *u_post*publish_count_q_reg*}]
set payload_sinks [get_cells -hierarchical -filter {full_name =~ *u_post*o_rgb_reg* || full_name =~ *u_post*active_bank_q_reg* || full_name =~ *u_post*active_count_pixel_q_reg*}]
if {[sizeof_collection $payload] == 0 || [sizeof_collection $payload_sinks] == 0} {error "Missing held-bbox audit endpoints"}
remove_clock_groups -asynchronous pixel_npu_async
set_max_delay 20.0 -from $payload -to $payload_sinks -group_path BBOX_SETTLING_AUDIT
set_min_delay 0.0 -from $payload -to $payload_sinks
redirect [file join $out_dir bbox_settling_audit.rpt] {report_timing -group BBOX_SETTLING_AUDIT -delay_type max -max_paths 10 -nworst 1}
set_clock_groups -name pixel_npu_async -asynchronous -group [get_clocks clk_pixel] -group [get_clocks clk_npu]
puts "BB_SYNTHESIS_COMPLETED: $topname eight_SRAMs_excluded"
exit 0

} bb_error]} {
    puts "BB_FATAL_SCRIPT_ERROR: $bb_error"
    exit 9
}
