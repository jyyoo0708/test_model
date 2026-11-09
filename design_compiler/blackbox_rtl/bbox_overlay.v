`default_nettype none

module bbox_overlay #(
    parameter integer p_sensor_width = 96,
    parameter integer p_sensor_height = 96,
    parameter integer p_max_boxes = 50
) (
    input wire clk_pixel,
    input wire clk_npu,
    input wire rst_pixel_n,
    input wire rst_npu_n,
    input wire i_result_valid,
    input wire [11:0] i_result_index,
    input wire [7:0] i_result_data,
    input wire i_result_done,
    input wire i_lut_we,
    input wire [10:0] i_lut_addr,
    input wire [15:0] i_lut_data,
    input wire i_cfg_we,
    input wire [7:0] i_cfg_addr,
    input wire [31:0] i_cfg_wdata,
    output reg [31:0] o_cfg_rdata,
    input wire i_vsync,
    input wire i_hsync,
    input wire [23:0] i_rgb,
    output reg o_vsync,
    output reg o_hsync,
    output reg [23:0] o_rgb,
    output wire o_busy,
    output reg o_box_valid,
    output reg [6:0] o_box_x1,
    output reg [6:0] o_box_y1,
    output reg [6:0] o_box_x2,
    output reg [6:0] o_box_y2,
    output reg [15:0] o_box_score,
    output reg o_boxes_done
);

localparam integer p_long_side = (p_sensor_width > p_sensor_height) ? p_sensor_width : p_sensor_height;
localparam integer p_resize_w_raw = (p_sensor_width * 96 + p_long_side/2) / p_long_side;
localparam integer p_resize_h_raw = (p_sensor_height * 96 + p_long_side/2) / p_long_side;
localparam integer p_resize_w = (p_resize_w_raw < 1) ? 1 : p_resize_w_raw;
localparam integer p_resize_h = (p_resize_h_raw < 1) ? 1 : p_resize_h_raw;
localparam integer p_pad_x = (96-p_resize_w)/2;
localparam integer p_pad_y = (96-p_resize_h)/2;
localparam [4:0] p_idle=5'd0, p_drain=5'd1, p_setup=5'd2,
    p_read_value=5'd3, p_get_value=5'd4, p_read_peak=5'd5,
    p_get_peak=5'd6, p_score=5'd7, p_select=5'd8,
    p_check=5'd9, p_nms=5'd10, p_accept=5'd11, p_complete=5'd12,
    p_scan_read=5'd13, p_scan_compare=5'd14, p_scan_finish=5'd15,
    p_min_commit=5'd16, p_max_commit=5'd17, p_next_cell=5'd18,
    p_nms_area=5'd19, p_nms_union=5'd20, p_nms_product=5'd21, p_nms_compare=5'd22;

wire [15:0] lut_data_q;
reg [11:0] lut_index_q;
reg lut_valid_q;
wire [10:0] lut_read_addr;
wire [2:0] result_channel;
wire [11:0] unused_result_channel;
reg [11:0] result_read_addr;
wire [15:0] result_read_q;

reg [15:0] threshold_q, iou_q;
reg [7:0] min_width_q, min_height_q;
reg [1:0] enable_q;
reg [15:0] run_threshold_q, run_iou_q;
reg [7:0] run_min_width_q, run_min_height_q;
reg run_nms_q;
reg [4:0] state_q;
reg [1:0] drain_q;
reg [9:0] cell_q;
reg [2:0] channel_q;
reg [3:0] neighbor_q;
reg peak_q;
reg [15:0] probability_q [0:5];
reg [5:0] candidate_count_q;
reg candidate_valid_q [0:p_max_boxes-1];
reg [15:0] candidate_score_q [0:p_max_boxes-1];
reg [9:0] candidate_position_q [0:p_max_boxes-1];
reg [14:0] candidate_x1_q [0:p_max_boxes-1];
reg [14:0] candidate_y1_q [0:p_max_boxes-1];
reg [14:0] candidate_x2_q [0:p_max_boxes-1];
reg [14:0] candidate_y2_q [0:p_max_boxes-1];
reg [15:0] current_score_q;
reg [14:0] current_x1_q, current_y1_q, current_x2_q, current_y2_q;
reg [5:0] kept_count_q, compare_q;
reg [14:0] kept_x1_q [0:p_max_boxes-1];
reg [14:0] kept_y1_q [0:p_max_boxes-1];
reg [14:0] kept_x2_q [0:p_max_boxes-1];
reg [14:0] kept_y2_q [0:p_max_boxes-1];
reg [15:0] box_sensor_x1_q [0:2*p_max_boxes-1];
reg [15:0] box_sensor_y1_q [0:2*p_max_boxes-1];
reg [15:0] box_sensor_x2_q [0:2*p_max_boxes-1];
reg [15:0] box_sensor_y2_q [0:2*p_max_boxes-1];
// Compute-domain ownership. Published payload remains fixed until pixel ack.
reg pending_q, write_bank_q;
reg publish_req_toggle_q, publish_bank_q;
reg [5:0] publish_count_q, active_count_q;
(* ASYNC_REG = "TRUE" *) reg publish_ack_meta_q, publish_ack_sync_q;
// Pixel-domain publication and rendering state.
(* ASYNC_REG = "TRUE" *) reg publish_req_meta_q, publish_req_sync_q;
(* ASYNC_REG = "TRUE" *) reg enable_pixel_meta_q, enable_pixel_sync_q;
reg publish_ack_toggle_q, active_bank_q;
reg [5:0] active_count_pixel_q;
wire publish_apply;
reg vsync_q, hsync_q;
reg [15:0] pixel_x_q, pixel_y_q;
reg border_hit;

integer cell_x, cell_y, neighbor_x, neighbor_y, neighbor_address;
reg [31:0] debug_right, debug_bottom;
reg neighbor_in_range;
reg [31:0] score_product;
reg [15:0] score_value;
integer center_x_fixed, center_y_fixed, width_fixed, height_fixed;
integer left_fixed, top_fixed, right_fixed, bottom_fixed;
reg [14:0] box_x1, box_y1, box_x2, box_y2;
// Two-clock candidate scan: array read -> scalar comparison. No chained reduction.
reg scan_min_q, scan_found_q, scan_valid_q;
reg [5:0] scan_index_q, scan_best_index_q;
reg [15:0] scan_score_q, scan_best_score_q;
reg [9:0] scan_position_q, scan_best_position_q;
reg [14:0] nms_iw_q, nms_ih_q, nms_cw_q, nms_ch_q, nms_pw_q, nms_ph_q;
integer intersection_w, intersection_h;
reg [29:0] intersection_area, current_area, previous_area;
reg [30:0] union_area;
reg [46:0] nms_left, nms_right;
reg nms_suppress;
reg dimensions_good;
integer accept_sensor_x1, accept_sensor_y1, accept_sensor_x2, accept_sensor_y2;
integer write_box_index, render_index, render_loop, reset_loop;
integer render_x, render_y;
reg render_bank;
reg [5:0] render_count;

assign unused_result_channel = i_result_index / 12'd576;
assign result_channel = unused_result_channel[2:0];
assign lut_read_addr = {result_channel,i_result_data};
assign o_busy = (state_q != p_idle) || pending_q;
assign publish_apply = i_vsync && !vsync_q &&
                       (publish_req_sync_q != publish_ack_toggle_q);

// Synchronize only control toggles/bits. Box banks and publish bank/count form
// a held bundled-data transfer; constrain their physical settling delay too.
always @(posedge clk_npu or negedge rst_npu_n) begin
    if (!rst_npu_n) begin
        publish_ack_meta_q <= 1'b0;
        publish_ack_sync_q <= 1'b0;
    end else begin
        publish_ack_meta_q <= publish_ack_toggle_q;
        publish_ack_sync_q <= publish_ack_meta_q;
    end
end
always @(posedge clk_pixel or negedge rst_pixel_n) begin
    if (!rst_pixel_n) begin
        publish_req_meta_q <= 1'b0;
        publish_req_sync_q <= 1'b0;
        enable_pixel_meta_q <= 1'b1;
        enable_pixel_sync_q <= 1'b1;
    end else begin
        publish_req_meta_q <= publish_req_toggle_q;
        publish_req_sync_q <= publish_req_meta_q;
        enable_pixel_meta_q <= enable_q[0];
        enable_pixel_sync_q <= enable_pixel_meta_q;
    end
end

// Memory instances are black boxes for DC; functional models are separate.
wire  mem_lut_i_write_enable;
wire [11-1:0] mem_lut_i_write_address;
wire [16-1:0] mem_lut_i_write_data;
wire  mem_lut_i_read_enable;
wire [11-1:0] mem_lut_i_read_address;
wire [16-1:0] mem_lut_o_read_data;
assign mem_lut_i_write_enable = i_lut_we && (i_lut_addr < 11'd1536);
assign mem_lut_i_write_address = i_lut_addr;
assign mem_lut_i_write_data = i_lut_data;
assign mem_lut_i_read_enable = i_result_valid && (i_result_index < 12'd3456);
assign mem_lut_i_read_address = lut_read_addr;
assign lut_data_q = mem_lut_o_read_data;
npu_sram_bb #(.p_depth(1536), .p_data_w(16), .p_addr_w(11)) u_mem_lut (
    .clk_write(clk_npu),
    .clk_read(clk_npu),
    .i_write_enable(mem_lut_i_write_enable),
    .i_write_address(mem_lut_i_write_address),
    .i_write_data(mem_lut_i_write_data),
    .i_read_enable(mem_lut_i_read_enable),
    .i_read_address(mem_lut_i_read_address),
    .o_read_data(mem_lut_o_read_data)
);

wire  mem_result_i_write_enable;
wire [12-1:0] mem_result_i_write_address;
wire [16-1:0] mem_result_i_write_data;
wire  mem_result_i_read_enable;
wire [12-1:0] mem_result_i_read_address;
wire [16-1:0] mem_result_o_read_data;
assign mem_result_i_write_enable = lut_valid_q;
assign mem_result_i_write_address = lut_index_q;
assign mem_result_i_write_data = lut_data_q;
assign mem_result_i_read_enable = 1'b1;
assign mem_result_i_read_address = result_read_addr;
assign result_read_q = mem_result_o_read_data;
npu_sram_bb #(.p_depth(3456), .p_data_w(16), .p_addr_w(12)) u_mem_result (
    .clk_write(clk_npu),
    .clk_read(clk_npu),
    .i_write_enable(mem_result_i_write_enable),
    .i_write_address(mem_result_i_write_address),
    .i_write_data(mem_result_i_write_data),
    .i_read_enable(mem_result_i_read_enable),
    .i_read_address(mem_result_i_read_address),
    .o_read_data(mem_result_o_read_data)
);

always @(posedge clk_npu or negedge rst_npu_n) begin
    if (!rst_npu_n) begin
        lut_valid_q <= 1'b0;
        lut_index_q <= 12'd0;
    end else begin
        lut_valid_q <= i_result_valid && (i_result_index < 12'd3456);
        lut_index_q <= i_result_index;
    end
end

always @(posedge clk_npu or negedge rst_npu_n) begin
    if (!rst_npu_n) begin
        threshold_q <= 16'd21299;
        iou_q <= 16'd32768;
        min_width_q <= 8'd8;
        min_height_q <= 8'd24;
        enable_q <= 2'b11;
    end else if (i_cfg_we) begin
        case (i_cfg_addr)
            8'h00: threshold_q <= i_cfg_wdata[15:0];
            8'h04: iou_q <= i_cfg_wdata[15:0];
            8'h08: begin min_width_q <= i_cfg_wdata[7:0]; min_height_q <= i_cfg_wdata[15:8]; end
            8'h0c: enable_q <= i_cfg_wdata[1:0];
            default: begin end
        endcase
    end
end
always @* begin
    o_cfg_rdata = 32'd0;
    case (i_cfg_addr)
        8'h00: o_cfg_rdata = {16'd0,threshold_q};
        8'h04: o_cfg_rdata = {16'd0,iou_q};
        8'h08: o_cfg_rdata = {16'd0,min_height_q,min_width_q};
        8'h0c: o_cfg_rdata = {30'd0,enable_q};
        8'h10: o_cfg_rdata = {24'd0,active_count_q,pending_q,(state_q != p_idle)};
        default: begin end
    endcase
end

always @* begin
    cell_x = {22'd0,cell_q} % 32'd24;
    cell_y = {22'd0,cell_q} / 32'd24;
    neighbor_x = cell_x + ({28'd0,neighbor_q} % 32'd3) - 32'd1;
    neighbor_y = cell_y + ({28'd0,neighbor_q} / 32'd3) - 32'd1;
    neighbor_in_range = (neighbor_x >= 0) && (neighbor_x < 24) && (neighbor_y >= 0) && (neighbor_y < 24);
    neighbor_address = neighbor_y*24 + neighbor_x;
    result_read_addr = 12'd0;
    if (state_q == p_read_value) begin
        result_read_addr = ({9'd0,channel_q} * 12'd576) + {2'd0,cell_q};
    end else if ((state_q == p_read_peak) && neighbor_in_range) begin
        result_read_addr = neighbor_address[11:0];
    end
end

// Full coordinate arithmetic in Q16, followed by conservative Q8 endpoints.
always @* begin
    score_product = probability_q[0] * probability_q[5];
    score_value = score_product[31:16];
    center_x_fixed = (cell_x * 262144) + ({16'd0,probability_q[1]} * 8) - 131072;
    center_y_fixed = (cell_y * 262144) + ({16'd0,probability_q[2]} * 8) - 131072;
    width_fixed = {16'd0,probability_q[3]} * 96;
    height_fixed = {16'd0,probability_q[4]} * 96;
    left_fixed = center_x_fixed - (width_fixed / 2);
    right_fixed = center_x_fixed + (width_fixed / 2);
    top_fixed = center_y_fixed - (height_fixed / 2);
    bottom_fixed = center_y_fixed + (height_fixed / 2);
    if (left_fixed < 0) left_fixed = 0;
    if (left_fixed > 6291456) left_fixed = 6291456;
    if (right_fixed < 0) right_fixed = 0;
    if (right_fixed > 6291456) right_fixed = 6291456;
    if (top_fixed < 0) top_fixed = 0;
    if (top_fixed > 6291456) top_fixed = 6291456;
    if (bottom_fixed < 0) bottom_fixed = 0;
    if (bottom_fixed > 6291456) bottom_fixed = 6291456;
    debug_right = ({17'd0,current_x2_q}+32'd255)/32'd256-32'd1;
    debug_bottom = ({17'd0,current_y2_q}+32'd255)/32'd256-32'd1;
    box_x1 = left_fixed[22:8];
    box_y1 = top_fixed[22:8];
    box_x2 = right_fixed[22:8];
    box_y2 = bottom_fixed[22:8];
end

always @* begin
    intersection_w = 0;
    intersection_h = 0;
    nms_suppress = nms_left > nms_right;
    dimensions_good = (({17'd0,current_x2_q}-{17'd0,current_x1_q}) >= ({24'd0,run_min_width_q} << 8)) &&
                      (({17'd0,current_y2_q}-{17'd0,current_y1_q}) >= ({24'd0,run_min_height_q} << 8)) &&
                      (current_x2_q > current_x1_q) && (current_y2_q > current_y1_q);
    if (compare_q < kept_count_q) begin
        intersection_w = ((current_x2_q < kept_x2_q[compare_q]) ? {17'd0,current_x2_q} : {17'd0,kept_x2_q[compare_q]}) -
                         ((current_x1_q > kept_x1_q[compare_q]) ? {17'd0,current_x1_q} : {17'd0,kept_x1_q[compare_q]});
        intersection_h = ((current_y2_q < kept_y2_q[compare_q]) ? {17'd0,current_y2_q} : {17'd0,kept_y2_q[compare_q]}) -
                         ((current_y1_q > kept_y1_q[compare_q]) ? {17'd0,current_y1_q} : {17'd0,kept_y1_q[compare_q]});
        if (intersection_w < 0) intersection_w = 0;
        if (intersection_h < 0) intersection_h = 0;
    end
end

// Letterbox inverse mapping uses constant divisors and supports arbitrary sensor aspect.
always @* begin
    accept_sensor_x1 = (($signed({17'd0,current_x1_q}) - p_pad_x*256) * p_sensor_width) / (p_resize_w*256);
    accept_sensor_y1 = (($signed({17'd0,current_y1_q}) - p_pad_y*256) * p_sensor_height) / (p_resize_h*256);
    accept_sensor_x2 = (($signed({17'd0,current_x2_q}) - p_pad_x*256) * p_sensor_width + p_resize_w*256-1) / (p_resize_w*256)-1;
    accept_sensor_y2 = (($signed({17'd0,current_y2_q}) - p_pad_y*256) * p_sensor_height + p_resize_h*256-1) / (p_resize_h*256)-1;
    if (accept_sensor_x1 < 0) accept_sensor_x1 = 0;
    if (accept_sensor_y1 < 0) accept_sensor_y1 = 0;
    if (accept_sensor_x2 < 0) accept_sensor_x2 = 0;
    if (accept_sensor_y2 < 0) accept_sensor_y2 = 0;
    if (accept_sensor_x1 >= p_sensor_width) accept_sensor_x1 = p_sensor_width-1;
    if (accept_sensor_y1 >= p_sensor_height) accept_sensor_y1 = p_sensor_height-1;
    if (accept_sensor_x2 >= p_sensor_width) accept_sensor_x2 = p_sensor_width-1;
    if (accept_sensor_y2 >= p_sensor_height) accept_sensor_y2 = p_sensor_height-1;
    write_box_index = (write_bank_q ? p_max_boxes : 0) + {26'd0,kept_count_q};
end

// Closely coupled counters, memory coordinates and transitions use one process.
always @(posedge clk_npu or negedge rst_npu_n) begin
    if (!rst_npu_n) begin
        state_q <= p_idle;
        drain_q <= 2'd0;
        cell_q <= 10'd0;
        channel_q <= 3'd0;
        neighbor_q <= 4'd0;
        peak_q <= 1'b0;
        scan_min_q <= 1'b0; scan_found_q <= 1'b0; scan_valid_q <= 1'b0;
        scan_index_q <= 6'd0; scan_best_index_q <= 6'd0;
        scan_score_q <= 16'd0; scan_best_score_q <= 16'd0;
        scan_position_q <= 10'd0; scan_best_position_q <= 10'd0;
        nms_iw_q <= 15'd0; nms_ih_q <= 15'd0;
        nms_cw_q <= 15'd0; nms_ch_q <= 15'd0; nms_pw_q <= 15'd0; nms_ph_q <= 15'd0;
        intersection_area <= 30'd0; current_area <= 30'd0; previous_area <= 30'd0;
        union_area <= 31'd0; nms_left <= 47'd0; nms_right <= 47'd0;
        candidate_count_q <= 6'd0;
        kept_count_q <= 6'd0;
        compare_q <= 6'd0;
        pending_q <= 1'b0;
        write_bank_q <= 1'b1;
        publish_req_toggle_q <= 1'b0;
        publish_bank_q <= 1'b1;
        publish_count_q <= 6'd0;
        active_count_q <= 6'd0;
        o_box_valid <= 1'b0;
        o_boxes_done <= 1'b0;
        o_box_x1 <= 7'd0; o_box_y1 <= 7'd0; o_box_x2 <= 7'd0; o_box_y2 <= 7'd0; o_box_score <= 16'd0;
        run_threshold_q <= 16'd21299; run_iou_q <= 16'd32768;
        run_min_width_q <= 8'd8; run_min_height_q <= 8'd24; run_nms_q <= 1'b1;
        current_score_q <= 16'd0;
        current_x1_q <= 15'd0; current_y1_q <= 15'd0; current_x2_q <= 15'd0; current_y2_q <= 15'd0;
        for (reset_loop=0; reset_loop<p_max_boxes; reset_loop=reset_loop+1) begin
            candidate_valid_q[reset_loop] <= 1'b0;
        end
    end else begin
        o_box_valid <= 1'b0;
        o_boxes_done <= 1'b0;
        if (pending_q && (publish_ack_sync_q == publish_req_toggle_q)) begin
            // Pixel has atomically moved to the published bank; only now may
            // compute reuse the formerly active bank for another result.
            pending_q <= 1'b0;
            write_bank_q <= !write_bank_q;
            active_count_q <= publish_count_q;
        end
        case (state_q)
            p_idle: begin
                if (i_result_done && !pending_q) begin
                    drain_q <= 2'd2;
                    run_threshold_q <= threshold_q;
                    run_iou_q <= iou_q;
                    run_min_width_q <= min_width_q;
                    run_min_height_q <= min_height_q;
                    run_nms_q <= enable_q[1];
                    state_q <= p_drain;
                end
            end
            p_drain: begin
                if (drain_q == 2'd0) state_q <= p_setup;
                else drain_q <= drain_q-2'd1;
            end
            p_setup: begin
                cell_q <= 10'd0;
                channel_q <= 3'd0;
                candidate_count_q <= 6'd0;
                kept_count_q <= 6'd0;
                for (reset_loop=0; reset_loop<p_max_boxes; reset_loop=reset_loop+1) candidate_valid_q[reset_loop] <= 1'b0;
                state_q <= p_read_value;
            end
            p_read_value: state_q <= p_get_value;
            p_get_value: begin
                probability_q[channel_q] <= result_read_q;
                if (channel_q == 3'd5) begin
                    neighbor_q <= 4'd0;
                    peak_q <= 1'b1;
                    state_q <= p_read_peak;
                end else begin
                    channel_q <= channel_q+3'd1;
                    state_q <= p_read_value;
                end
            end
            p_read_peak: state_q <= p_get_peak;
            p_get_peak: begin
                if (neighbor_in_range && (result_read_q > probability_q[0])) peak_q <= 1'b0;
                if (neighbor_q == 4'd8) state_q <= p_score;
                else begin neighbor_q <= neighbor_q+4'd1; state_q <= p_read_peak; end
            end
            p_score: begin
                if (peak_q && (score_value >= run_threshold_q)) begin
                    if ({26'd0,candidate_count_q} < p_max_boxes) begin
                        candidate_valid_q[candidate_count_q] <= 1'b1;
                        candidate_score_q[candidate_count_q] <= score_value;
                        candidate_position_q[candidate_count_q] <= cell_q;
                        candidate_x1_q[candidate_count_q] <= box_x1;
                        candidate_y1_q[candidate_count_q] <= box_y1;
                        candidate_x2_q[candidate_count_q] <= box_x2;
                        candidate_y2_q[candidate_count_q] <= box_y2;
                        candidate_count_q <= candidate_count_q+6'd1;
                        state_q <= p_next_cell;
                    end else begin
                        scan_min_q <= 1'b1;
                        scan_found_q <= 1'b0;
                        scan_index_q <= 6'd0;
                        state_q <= p_scan_read;
                    end
                end else state_q <= p_next_cell;
            end
            p_next_cell: begin
                if (cell_q == 10'd575) state_q <= p_select;
                else begin cell_q <= cell_q+10'd1; channel_q <= 3'd0; state_q <= p_read_value; end
            end
            p_select: begin
                if ({26'd0,kept_count_q} == p_max_boxes) state_q <= p_complete;
                else begin
                    scan_min_q <= 1'b0;
                    scan_found_q <= 1'b0;
                    scan_index_q <= 6'd0;
                    state_q <= p_scan_read;
                end
            end
            p_scan_read: begin
                scan_valid_q <= candidate_valid_q[scan_index_q];
                scan_score_q <= candidate_score_q[scan_index_q];
                scan_position_q <= candidate_position_q[scan_index_q];
                state_q <= p_scan_compare;
            end
            p_scan_compare: begin
                // Replacement: minimum score, latest raster on equality.
                // Output order: maximum score, earliest raster on equality.
                if (scan_valid_q && (!scan_found_q ||
                    (scan_min_q && ((scan_score_q < scan_best_score_q) ||
                     ((scan_score_q == scan_best_score_q) && (scan_position_q > scan_best_position_q)))) ||
                    (!scan_min_q && ((scan_score_q > scan_best_score_q) ||
                     ((scan_score_q == scan_best_score_q) && (scan_position_q < scan_best_position_q)))))) begin
                    scan_found_q <= 1'b1;
                    scan_best_index_q <= scan_index_q;
                    scan_best_score_q <= scan_score_q;
                    scan_best_position_q <= scan_position_q;
                end
                if ({26'd0,scan_index_q}+32'd1 == p_max_boxes) state_q <= p_scan_finish;
                else begin scan_index_q <= scan_index_q+6'd1; state_q <= p_scan_read; end
            end
            p_scan_finish: begin
                if (scan_min_q) state_q <= p_min_commit;
                else state_q <= p_max_commit;
            end
            p_min_commit: begin
                if (scan_found_q && (score_value > scan_best_score_q)) begin
                    candidate_score_q[scan_best_index_q] <= score_value;
                    candidate_position_q[scan_best_index_q] <= cell_q;
                    candidate_x1_q[scan_best_index_q] <= box_x1;
                    candidate_y1_q[scan_best_index_q] <= box_y1;
                    candidate_x2_q[scan_best_index_q] <= box_x2;
                    candidate_y2_q[scan_best_index_q] <= box_y2;
                end
                state_q <= p_next_cell;
            end
            p_max_commit: begin
                if (!scan_found_q) state_q <= p_complete;
                else begin
                    current_score_q <= scan_best_score_q;
                    current_x1_q <= candidate_x1_q[scan_best_index_q]; current_y1_q <= candidate_y1_q[scan_best_index_q];
                    current_x2_q <= candidate_x2_q[scan_best_index_q]; current_y2_q <= candidate_y2_q[scan_best_index_q];
                    candidate_valid_q[scan_best_index_q] <= 1'b0;
                    compare_q <= 6'd0;
                    state_q <= p_check;
                end
            end
            p_check: begin
                if (!dimensions_good) state_q <= p_select;
                else if (!run_nms_q || (kept_count_q == 6'd0)) state_q <= p_accept;
                else state_q <= p_nms;
            end
            p_nms: begin
                nms_iw_q <= intersection_w[14:0]; nms_ih_q <= intersection_h[14:0];
                nms_cw_q <= current_x2_q-current_x1_q; nms_ch_q <= current_y2_q-current_y1_q;
                nms_pw_q <= kept_x2_q[compare_q]-kept_x1_q[compare_q];
                nms_ph_q <= kept_y2_q[compare_q]-kept_y1_q[compare_q];
                state_q <= p_nms_area;
            end
            p_nms_area: begin
                intersection_area <= nms_iw_q*nms_ih_q;
                current_area <= nms_cw_q*nms_ch_q;
                previous_area <= nms_pw_q*nms_ph_q;
                state_q <= p_nms_union;
            end
            p_nms_union: begin
                union_area <= {1'b0,current_area}+{1'b0,previous_area}-{1'b0,intersection_area};
                state_q <= p_nms_product;
            end
            p_nms_product: begin
                nms_left <= {1'b0,intersection_area,16'd0};
                nms_right <= union_area*run_iou_q;
                state_q <= p_nms_compare;
            end
            p_nms_compare: begin
                if (nms_suppress) state_q <= p_select;
                else if (compare_q+6'd1 == kept_count_q) state_q <= p_accept;
                else begin compare_q <= compare_q+6'd1; state_q <= p_nms; end
            end
            p_accept: begin
                kept_x1_q[kept_count_q] <= current_x1_q; kept_y1_q[kept_count_q] <= current_y1_q;
                kept_x2_q[kept_count_q] <= current_x2_q; kept_y2_q[kept_count_q] <= current_y2_q;
                box_sensor_x1_q[write_box_index] <= accept_sensor_x1[15:0];
                box_sensor_y1_q[write_box_index] <= accept_sensor_y1[15:0];
                box_sensor_x2_q[write_box_index] <= accept_sensor_x2[15:0];
                box_sensor_y2_q[write_box_index] <= accept_sensor_y2[15:0];
                o_box_x1 <= current_x1_q[14:8]; o_box_y1 <= current_y1_q[14:8];
                o_box_x2 <= debug_right[6:0];
                o_box_y2 <= debug_bottom[6:0];
                o_box_score <= current_score_q;
                o_box_valid <= 1'b1;
                kept_count_q <= kept_count_q+6'd1;
                state_q <= p_select;
            end
            p_complete: begin
                publish_bank_q <= write_bank_q;
                publish_count_q <= kept_count_q;
                publish_req_toggle_q <= !publish_req_toggle_q;
                pending_q <= 1'b1;
                o_boxes_done <= 1'b1;
                state_q <= p_idle;
            end
            default: state_q <= p_idle;
        endcase
    end
end

// Render from a stable published bank. On the publication edge the combinational
// selector also sees the new bank, including a valid first pixel at VSYNC rise.
always @* begin
    render_bank = active_bank_q;
    render_count = active_count_pixel_q;
    if (publish_apply) begin
        render_bank = publish_bank_q;
        render_count = publish_count_q;
    end
    render_x = {16'd0,pixel_x_q};
    render_y = {16'd0,pixel_y_q};
    if (i_vsync && !vsync_q) begin render_x=0; render_y=0; end
    border_hit = 1'b0;
    render_index = 0;
    for (render_loop=0; render_loop<p_max_boxes; render_loop=render_loop+1) begin
        render_index = (render_bank ? p_max_boxes : 0)+render_loop;
        if ((render_loop < {26'd0,render_count}) && (render_x >= {16'd0,box_sensor_x1_q[render_index]}) &&
            (render_x <= {16'd0,box_sensor_x2_q[render_index]}) && (render_y >= {16'd0,box_sensor_y1_q[render_index]}) &&
            (render_y <= {16'd0,box_sensor_y2_q[render_index]}) &&
            ((render_x == {16'd0,box_sensor_x1_q[render_index]}) || (render_x == {16'd0,box_sensor_x2_q[render_index]}) ||
             (render_y == {16'd0,box_sensor_y1_q[render_index]}) || (render_y == {16'd0,box_sensor_y2_q[render_index]}))) begin
            border_hit = 1'b1;
        end
    end
end
always @(posedge clk_pixel or negedge rst_pixel_n) begin
    if (!rst_pixel_n) begin
        publish_ack_toggle_q <= 1'b0;
        active_bank_q <= 1'b0;
        active_count_pixel_q <= 6'd0;
        vsync_q <= 1'b0; hsync_q <= 1'b0;
        pixel_x_q <= 16'd0; pixel_y_q <= 16'd0;
        o_vsync <= 1'b0; o_hsync <= 1'b0; o_rgb <= 24'd0;
    end else begin
        if (publish_apply) begin
            active_bank_q <= publish_bank_q;
            active_count_pixel_q <= publish_count_q;
            publish_ack_toggle_q <= publish_req_sync_q;
        end
        vsync_q <= i_vsync;
        hsync_q <= i_hsync;
        o_vsync <= i_vsync;
        o_hsync <= i_hsync;
        o_rgb <= (i_vsync && i_hsync && enable_pixel_sync_q && border_hit) ? 24'hff0000 : i_rgb;
        if (!i_vsync) begin pixel_x_q <= 16'd0; pixel_y_q <= 16'd0; end
        else if (!vsync_q) begin pixel_x_q <= i_hsync ? 16'd1 : 16'd0; pixel_y_q <= 16'd0; end
        else if (i_hsync) pixel_x_q <= pixel_x_q+16'd1;
        else if (hsync_q) begin pixel_x_q <= 16'd0; pixel_y_q <= pixel_y_q+16'd1; end
    end
end

endmodule
`default_nettype wire
