`default_nettype none
(* black_box = "true", syn_black_box = 1 *)
module npu_sram_bb #(
    parameter integer p_depth = 1024,
    parameter integer p_data_w = 8,
    parameter integer p_addr_w = 10
) (
    input wire clk_write,
    input wire clk_read,
    input wire i_write_enable,
    input wire [p_addr_w-1:0] i_write_address,
    input wire [p_data_w-1:0] i_write_data,
    input wire i_read_enable,
    input wire [p_addr_w-1:0] i_read_address,
    output wire [p_data_w-1:0] o_read_data
);
// Intentionally no data/timing/area model. Preserve all eight instances in DC.
endmodule
`default_nettype wire
