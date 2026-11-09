set run_root [file normalize $env(NPU_DC_ROOT)]
set out_dir [file normalize $env(NPU_DC_OUT)]
set rtl_dir [file join $run_root rtl_snapshot]
set topname person_detector_top
set stdcell /tools/lib/ln28lpp/primitive/ln28lpp_sc_9t_base_lvt_c130_FE-Common_sec190802_0202/LIBERTY/sc9_cmos28lpp_base_lvt_c34_ss_nominal_max_0p950v_m25c_sadhm.db
if {![file exists $stdcell]} {error "Missing standard-cell DB: $stdcell"}
file mkdir $out_dir
define_design_lib WORK -path [file join $out_dir work]
set search_path [concat [list $rtl_dir /tools/synopsys/syn/I-2013.12-SP4/libraries/syn] $search_path]
set target_library [list $stdcell]
set synthetic_library [list dw_foundation.sldb]
set link_library [concat [list * dw_foundation.sldb] $target_library]
set_svf [file join $out_dir ${topname}.svf]
catch {set_host_options -max_cores 4}
set sources [list]
foreach name {dvp_capture.v npu_core.v bbox_overlay.v person_detector_top.v} {
    lappend sources [file join $rtl_dir $name]
}
puts "NPU_STAGE: analyze"
if {![analyze -format verilog $sources]} {puts "NPU_FAILED: analyze"; exit 2}
puts "NPU_STAGE: elaborate (original full memory arrays, no black boxes)"
if {![elaborate $topname]} {puts "NPU_FAILED: elaborate"; exit 3}
current_design $topname
puts "NPU_STAGE: link"
if {![link]} {puts "NPU_FAILED: link"; exit 4}
redirect [file join $out_dir check_design_pre.rpt] {check_design}
redirect [file join $out_dir resources_pre.rpt] {report_resources -hierarchy}
redirect [file join $out_dir references_pre.rpt] {report_reference -hierarchy}
write -format ddc -hierarchy -output [file join $out_dir ${topname}_elaborated.ddc]
create_clock -name clk -period [expr {1000.0/27.0}] [get_ports clk]
set_clock_uncertainty -setup 0.1 [get_clocks clk]
set_clock_uncertainty -hold 0.05 [get_clocks clk]
set_clock_transition 0.1 [get_clocks clk]
set data_inputs [remove_from_collection [all_inputs] [get_ports {clk rst_n}]]
set_input_delay -max 1.0 -clock clk $data_inputs
set_input_delay -min 0.0 -clock clk $data_inputs
set_input_transition 0.2 $data_inputs
set_output_delay -max 1.0 -clock clk [all_outputs]
set_output_delay -min 0.0 -clock clk [all_outputs]
set_load 0.01 [all_outputs]
# Asynchronous reset assertion/release paths require a separate board/reset audit.
set_false_path -from [get_ports rst_n]
set_max_fanout 20 [current_design]
redirect [file join $out_dir check_timing_pre.rpt] {check_timing}
write_sdc [file join $out_dir ${topname}.sdc]
puts "NPU_STAGE: compile"
if {![compile]} {puts "NPU_FAILED: compile"; exit 6}
puts "NPU_STAGE: reports"
change_names -rules verilog -hierarchy
set verilogout_no_tri true
write -format verilog -hierarchy -output [file join $out_dir ${topname}_syn.v]
write -format ddc -hierarchy -output [file join $out_dir ${topname}.ddc]
write_sdc [file join $out_dir ${topname}.sdc]
redirect [file join $out_dir timing_setup.rpt] {report_timing -delay_type max -max_paths 20 -nworst 1}
redirect [file join $out_dir timing_hold.rpt] {report_timing -delay_type min -max_paths 20 -nworst 1}
redirect [file join $out_dir area.rpt] {report_area -hierarchy}
redirect [file join $out_dir constraints.rpt] {report_constraint -all_violators}
redirect [file join $out_dir qor.rpt] {report_qor}
redirect [file join $out_dir references.rpt] {report_reference -hierarchy}
redirect [file join $out_dir design.rpt] {report_design}
redirect [file join $out_dir check_design_post.rpt] {check_design}
redirect [file join $out_dir check_timing_post.rpt] {check_timing}
puts "NPU_SYNTHESIS_COMPLETED: $topname"
exit 0
