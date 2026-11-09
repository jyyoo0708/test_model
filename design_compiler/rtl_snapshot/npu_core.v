`default_nettype none
// Scalar, output-stationary complete student detector. All SRAM reads synchronous.
module npu_core #(
    parameter p_activation_depth = 108288,
    parameter p_weight_depth = 89820,
    parameter p_parameter_depth = 274,
    parameter p_descriptor_depth = 208,
    parameter p_output_base = 104832
) (
    input wire clk,
    input wire rst_n,
    input wire i_frame_start,
    input wire i_pixel_we,
    input wire [13:0] i_pixel_addr,
    input wire signed [7:0] i_pixel_data,
    input wire i_load_we,
    input wire [2:0] i_load_space,
    input wire [17:0] i_load_addr,
    input wire [31:0] i_load_data,
    output wire o_busy,
    output reg o_done,
    output reg o_result_valid,
    output reg [11:0] o_result_index,
    output reg signed [7:0] o_result_data
);
localparam [3:0] p_idle=4'd0, p_desc_request=4'd1, p_desc_use=4'd2,
    p_parameter_request=4'd3, p_parameter_use=4'd4,
    p_mac_request=4'd5, p_mac_use=4'd6, p_requant=4'd7,
    p_stream_request=4'd8, p_stream_use=4'd9, p_done=4'd10;
localparam p_activation_aw = $clog2(p_activation_depth);
localparam p_weight_aw = $clog2(p_weight_depth);
localparam p_parameter_aw = $clog2(p_parameter_depth);
localparam p_descriptor_aw = $clog2(p_descriptor_depth);
(* ram_style = "block" *) reg signed [7:0] activation_mem [0:p_activation_depth-1];
(* ram_style = "block" *) reg signed [7:0] weight_mem [0:p_weight_depth-1];
(* ram_style = "block" *) reg signed [31:0] bias_mem [0:p_parameter_depth-1];
(* ram_style = "block" *) reg signed [31:0] multiplier_mem [0:p_parameter_depth-1];
(* ram_style = "block" *) reg [31:0] descriptor_mem [0:p_descriptor_depth-1];
reg [3:0] state_q;
reg [3:0] layer_q, field_q;
reg [31:0] descriptor_q [0:15];
reg [31:0] descriptor_data_q;
reg signed [7:0] activation_data_q, weight_data_q;
reg activation_padding_q;
reg signed [31:0] bias_data_q, multiplier_data_q;
reg signed [31:0] accumulator_q;
reg [31:0] ox_q, oy_q, oc_q, ic_q, kx_q, ky_q;
reg [11:0] stream_index_q;
wire [31:0] input_channels;
wire signed [31:0] ix, iy;
wire padding_pixel;
wire [31:0] source_x, source_y, source_width, source_height;
wire [31:0] activation_addr, weight_addr, output_addr, parameter_addr;
wire [31:0] descriptor_addr;
wire signed [15:0] product;
wire signed [31:0] product_extended;
wire signed [63:0] requant_product, requant_value;
wire signed [7:0] requant_byte;
wire last_kernel, last_output;
wire activation_read_enable, activation_write_enable;
wire [p_activation_aw-1:0] activation_read_addr, activation_write_addr;
wire signed [7:0] activation_write_data;

assign o_busy = state_q != p_idle;
assign input_channels = descriptor_q[7] + descriptor_q[8];
assign ix = $signed(ox_q * descriptor_q[13] + kx_q) - $signed(descriptor_q[12] >> 1);
assign iy = $signed(oy_q * descriptor_q[13] + ky_q) - $signed(descriptor_q[12] >> 1);
assign padding_pixel = (ix < 0) || (iy < 0) ||
    (ix >= $signed(descriptor_q[5])) || (iy >= $signed(descriptor_q[6]));
assign source_x = (ic_q < descriptor_q[7] && descriptor_q[14] != 0) ?
    ($unsigned(ix) >> 1) : $unsigned(ix);
assign source_y = (ic_q < descriptor_q[7] && descriptor_q[14] != 0) ?
    ($unsigned(iy) >> 1) : $unsigned(iy);
assign source_width = (ic_q < descriptor_q[7] && descriptor_q[14] != 0) ?
    (descriptor_q[5] >> 1) : descriptor_q[5];
assign source_height = (ic_q < descriptor_q[7] && descriptor_q[14] != 0) ?
    (descriptor_q[6] >> 1) : descriptor_q[6];
assign activation_addr = (ic_q < descriptor_q[7] ? descriptor_q[0] : descriptor_q[1]) +
    ((ic_q < descriptor_q[7] ? ic_q : ic_q-descriptor_q[7]) * source_height + source_y) * source_width + source_x;
assign weight_addr = descriptor_q[3] +
    ((oc_q * input_channels + ic_q) * descriptor_q[12] + ky_q) * descriptor_q[12] + kx_q;
assign output_addr = descriptor_q[2] + (oc_q * descriptor_q[10] + oy_q)*descriptor_q[9] + ox_q;
assign parameter_addr = descriptor_q[4] + oc_q;
assign descriptor_addr = {24'd0,layer_q,field_q};
assign product = activation_padding_q ? 16'sd0 : activation_data_q * weight_data_q;
assign product_extended = {{16{product[15]}},product};
assign requant_product = accumulator_q * multiplier_data_q;
assign requant_value = (requant_product + 64'sd8388608) >>> 24;
assign requant_byte = (requant_value > 64'sd127) ? 8'sd127 :
    (descriptor_q[15] != 0 && requant_value < 0) ? 8'sd0 :
    (requant_value < -64'sd128) ? -8'sd128 : requant_value[7:0];
assign last_kernel = kx_q+32'd1 == descriptor_q[12] &&
    ky_q+32'd1 == descriptor_q[12] && ic_q+32'd1 == input_channels;
assign last_output = ox_q+32'd1 == descriptor_q[9] &&
    oy_q+32'd1 == descriptor_q[10] && oc_q+32'd1 == descriptor_q[11];
assign activation_read_enable = state_q == p_mac_request || state_q == p_stream_request;
assign activation_read_addr = state_q == p_stream_request ?
    p_output_base[p_activation_aw-1:0] + {{(p_activation_aw-12){1'b0}},stream_index_q} :
    padding_pixel ? {p_activation_aw{1'b0}} : activation_addr[p_activation_aw-1:0];
assign activation_write_enable = state_q == p_requant ||
    (!o_busy && i_pixel_we && i_pixel_addr < 14'd9216);
assign activation_write_addr = state_q == p_requant ? output_addr[p_activation_aw-1:0] :
    {{(p_activation_aw-14){1'b0}},i_pixel_addr};
assign activation_write_data = state_q == p_requant ? requant_byte : i_pixel_data;

// Each inferred RAM uses synchronous registered read and synchronous write.
// No reset loop: control reset must not turn SRAM into flip-flops.
always @(posedge clk) begin
    if (!o_busy && i_load_we) begin
        case (i_load_space)
            3'd0: if (i_load_addr < p_weight_depth) weight_mem[i_load_addr[p_weight_aw-1:0]] <= i_load_data[7:0];
            3'd1: if (i_load_addr < p_parameter_depth) bias_mem[i_load_addr[p_parameter_aw-1:0]] <= i_load_data;
            3'd2: if (i_load_addr < p_descriptor_depth) descriptor_mem[i_load_addr[p_descriptor_aw-1:0]] <= i_load_data;
            3'd3: if (i_load_addr < p_parameter_depth) multiplier_mem[i_load_addr[p_parameter_aw-1:0]] <= i_load_data;
            default: begin end
        endcase
    end
    if (state_q == p_desc_request) begin
        descriptor_data_q <= descriptor_mem[descriptor_addr[p_descriptor_aw-1:0]];
    end
    if (state_q == p_parameter_request) begin
        bias_data_q <= bias_mem[parameter_addr[p_parameter_aw-1:0]];
        multiplier_data_q <= multiplier_mem[parameter_addr[p_parameter_aw-1:0]];
    end
end

// A single read address mux and a single write address mux make this a
// simple-dual-port BRAM: portA synchronous read, portB synchronous write.
always @(posedge clk) begin
    if (activation_read_enable) begin
        activation_data_q <= activation_mem[activation_read_addr];
    end
    if (activation_write_enable) begin
        activation_mem[activation_write_addr] <= activation_write_data;
    end
end
always @(posedge clk) begin
    if (state_q == p_mac_request) begin
        activation_padding_q <= padding_pixel;
        weight_data_q <= weight_mem[weight_addr[p_weight_aw-1:0]];
    end
end

// Closely coupled counters and controller retain an explicit cycle schedule.
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state_q <= p_idle;
        layer_q <= 4'd0;
        field_q <= 4'd0;
        ox_q <= 32'd0;
        oy_q <= 32'd0;
        oc_q <= 32'd0;
        ic_q <= 32'd0;
        kx_q <= 32'd0;
        ky_q <= 32'd0;
        stream_index_q <= 12'd0;
        accumulator_q <= 32'sd0;
        o_done <= 1'b0;
        o_result_valid <= 1'b0;
        o_result_index <= 12'd0;
        o_result_data <= 8'sd0;
    end else begin
        o_done <= 1'b0;
        o_result_valid <= 1'b0;
        case (state_q)
            p_idle: begin
                if (i_frame_start) begin
                    layer_q <= 4'd0;
                    field_q <= 4'd0;
                    state_q <= p_desc_request;
                end
            end
            p_desc_request: state_q <= p_desc_use;
            p_desc_use: begin
                descriptor_q[field_q] <= descriptor_data_q;
                if (field_q == 4'd15) begin
                    ox_q <= 32'd0;
                    oy_q <= 32'd0;
                    oc_q <= 32'd0;
                    state_q <= p_parameter_request;
                end else begin
                    field_q <= field_q + 4'd1;
                    state_q <= p_desc_request;
                end
            end
            p_parameter_request: state_q <= p_parameter_use;
            p_parameter_use: begin
                accumulator_q <= bias_data_q;
                ic_q <= 32'd0;
                kx_q <= 32'd0;
                ky_q <= 32'd0;
                state_q <= p_mac_request;
            end
            p_mac_request: state_q <= p_mac_use;
            p_mac_use: begin
                accumulator_q <= accumulator_q + product_extended;
                if (last_kernel) begin
                    state_q <= p_requant;
                end else begin
                    if (kx_q+32'd1 == descriptor_q[12]) begin
                        kx_q <= 32'd0;
                        if (ky_q+32'd1 == descriptor_q[12]) begin
                            ky_q <= 32'd0;
                            ic_q <= ic_q + 32'd1;
                        end else begin
                            ky_q <= ky_q + 32'd1;
                        end
                    end else begin
                        kx_q <= kx_q + 32'd1;
                    end
                    state_q <= p_mac_request;
                end
            end
            p_requant: begin
                if (last_output) begin
                    if (layer_q == 4'd12) begin
                        stream_index_q <= 12'd0;
                        state_q <= p_stream_request;
                    end else begin
                        layer_q <= layer_q + 4'd1;
                        field_q <= 4'd0;
                        state_q <= p_desc_request;
                    end
                end else begin
                    if (ox_q+32'd1 == descriptor_q[9]) begin
                        ox_q <= 32'd0;
                        if (oy_q+32'd1 == descriptor_q[10]) begin
                            oy_q <= 32'd0;
                            oc_q <= oc_q + 32'd1;
                        end else begin
                            oy_q <= oy_q + 32'd1;
                        end
                    end else begin
                        ox_q <= ox_q + 32'd1;
                    end
                    state_q <= p_parameter_request;
                end
            end
            p_stream_request: state_q <= p_stream_use;
            p_stream_use: begin
                o_result_valid <= 1'b1;
                o_result_index <= stream_index_q;
                o_result_data <= activation_data_q;
                if (stream_index_q == 12'd3455) begin
                    state_q <= p_done;
                end else begin
                    stream_index_q <= stream_index_q + 12'd1;
                    state_q <= p_stream_request;
                end
            end
            p_done: begin
                o_done <= 1'b1;
                state_q <= p_idle;
            end
            default: state_q <= p_idle;
        endcase
    end
end
endmodule
`default_nettype wire
