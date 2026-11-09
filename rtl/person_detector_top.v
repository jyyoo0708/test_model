`default_nettype none
module person_detector_top #(
    parameter integer p_sensor_width=96,
    parameter integer p_sensor_height=96,
    parameter integer p_input_type=0
) (
    input wire  clk_pixel,
    input wire  clk_npu,
    input wire  rst_n,
    input wire  i_vsync,
    input wire  i_hsync,
    input wire [23:0] i_data,
    input wire  i_load_we,
    input wire [2:0] i_load_space,
    input wire [17:0] i_load_addr,
    input wire [31:0] i_load_data,
    input wire  i_cfg_we,
    input wire [7:0] i_cfg_addr,
    input wire [31:0] i_cfg_wdata,
    output wire [31:0] o_cfg_rdata,
    output wire  o_ready,
    output wire  o_busy,
    output wire  o_frame_error,
    output wire  o_vsync,
    output wire  o_hsync,
    output wire [23:0] o_rgb,
    output wire  o_result_valid,
    output wire [11:0] o_result_index,
    output wire [7:0] o_result_data,
    output wire  o_npu_done,
    output wire  o_box_valid,
    output wire [6:0] o_box_x1,
    output wire [6:0] o_box_y1,
    output wire [6:0] o_box_x2,
    output wire [6:0] o_box_y2,
    output wire [15:0] o_box_score,
    output wire  o_boxes_done
);

    wire  capture_i_vsync;
    wire  capture_i_hsync;
    wire [23:0] capture_i_data;
    wire  capture_i_inference_done;
    wire  capture_o_pixel_we;
    wire [13:0] capture_o_pixel_addr;
    wire [7:0] capture_o_pixel_data;
    wire  capture_o_frame_start;
    wire  capture_o_ready;
    wire  capture_o_frame_error;
    wire [23:0] capture_o_bypass_rgb;

    wire  npu_i_frame_start;
    wire  npu_i_pixel_we;
    wire [13:0] npu_i_pixel_addr;
    wire [7:0] npu_i_pixel_data;
    wire  npu_i_load_we;
    wire [2:0] npu_i_load_space;
    wire [17:0] npu_i_load_addr;
    wire [31:0] npu_i_load_data;
    wire  npu_o_busy;
    wire  npu_o_done;
    wire  npu_o_result_valid;
    wire [11:0] npu_o_result_index;
    wire [7:0] npu_o_result_data;

    wire  post_i_result_valid;
    wire [11:0] post_i_result_index;
    wire [7:0] post_i_result_data;
    wire  post_i_result_done;
    wire  post_i_vsync;
    wire  post_i_hsync;
    wire [23:0] post_i_rgb;
    wire  post_i_cfg_we;
    wire [7:0] post_i_cfg_addr;
    wire [31:0] post_i_cfg_wdata;
    wire [31:0] post_o_cfg_rdata;
    wire  post_i_lut_we;
    wire [10:0] post_i_lut_addr;
    wire [15:0] post_i_lut_data;
    wire  post_o_vsync;
    wire  post_o_hsync;
    wire [23:0] post_o_rgb;
    wire  post_o_box_valid;
    wire [6:0] post_o_box_x1;
    wire [6:0] post_o_box_y1;
    wire [6:0] post_o_box_x2;
    wire [6:0] post_o_box_y2;
    wire [15:0] post_o_box_score;
    wire  post_o_boxes_done;
    wire  post_o_busy;


    // Common reset assertion, independent two-edge synchronous release.
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) reg [1:0] pixel_reset_sync_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) reg [1:0] npu_reset_sync_q;
    wire rst_pixel_n, rst_npu_n;
    reg frame_req_toggle_q, boxes_done_toggle_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) reg frame_req_meta_q, frame_req_sync_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) reg boxes_done_meta_q, boxes_done_sync_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) reg npu_busy_meta_q, npu_busy_sync_q;
    (* ASYNC_REG = "TRUE", SHREG_EXTRACT = "NO" *) reg post_busy_meta_q, post_busy_sync_q;
    reg frame_req_seen_q, boxes_done_seen_q;
    assign rst_pixel_n = pixel_reset_sync_q[1];
    assign rst_npu_n = npu_reset_sync_q[1];

    always @(posedge clk_pixel or negedge rst_n) begin
        if (!rst_n) begin
            pixel_reset_sync_q <= 2'b00;
        end else begin
            pixel_reset_sync_q <= {pixel_reset_sync_q[0],1'b1};
        end
    end
    always @(posedge clk_npu or negedge rst_n) begin
        if (!rst_n) begin
            npu_reset_sync_q <= 2'b00;
        end else begin
            npu_reset_sync_q <= {npu_reset_sync_q[0],1'b1};
        end
    end
    // Pixel owns the entire input frame until postprocessing completion returns.
    // The source memory remains stable while the NPU copies and consumes it.
    always @(posedge clk_pixel or negedge rst_pixel_n) begin
        if (!rst_pixel_n) begin
            frame_req_toggle_q <= 1'b0;
            boxes_done_meta_q <= 1'b0;
            boxes_done_sync_q <= 1'b0;
            boxes_done_seen_q <= 1'b0;
            npu_busy_meta_q <= 1'b0;
            npu_busy_sync_q <= 1'b0;
            post_busy_meta_q <= 1'b0;
            post_busy_sync_q <= 1'b0;
        end else begin
            if (capture_o_frame_start) begin
                frame_req_toggle_q <= !frame_req_toggle_q;
            end
            boxes_done_meta_q <= boxes_done_toggle_q;
            boxes_done_sync_q <= boxes_done_meta_q;
            boxes_done_seen_q <= boxes_done_sync_q;
            npu_busy_meta_q <= npu_o_busy;
            npu_busy_sync_q <= npu_busy_meta_q;
            post_busy_meta_q <= post_o_busy;
            post_busy_sync_q <= post_busy_meta_q;
        end
    end
    always @(posedge clk_npu or negedge rst_npu_n) begin
        if (!rst_npu_n) begin
            frame_req_meta_q <= 1'b0;
            frame_req_sync_q <= 1'b0;
            frame_req_seen_q <= 1'b0;
            boxes_done_toggle_q <= 1'b0;
        end else begin
            frame_req_meta_q <= frame_req_toggle_q;
            frame_req_sync_q <= frame_req_meta_q;
            frame_req_seen_q <= frame_req_sync_q;
            if (post_o_boxes_done) begin
                boxes_done_toggle_q <= !boxes_done_toggle_q;
            end
        end
    end

    assign capture_i_vsync = i_vsync;
    assign capture_i_hsync = i_hsync;
    assign capture_i_data = i_data;
    assign capture_i_inference_done = boxes_done_sync_q ^ boxes_done_seen_q;

    assign npu_i_frame_start = frame_req_sync_q ^ frame_req_seen_q;
    assign npu_i_pixel_we = capture_o_pixel_we;
    assign npu_i_pixel_addr = capture_o_pixel_addr;
    assign npu_i_pixel_data = capture_o_pixel_data;
    assign npu_i_load_we = i_load_we && (i_load_space < 3'd4);
    assign npu_i_load_space = i_load_space;
    assign npu_i_load_addr = i_load_addr;
    assign npu_i_load_data = i_load_data;

    assign post_i_result_valid = npu_o_result_valid;
    assign post_i_result_index = npu_o_result_index;
    assign post_i_result_data = npu_o_result_data;
    assign post_i_result_done = npu_o_done;
    assign post_i_vsync = i_vsync;
    assign post_i_hsync = i_hsync;
    assign post_i_rgb = capture_o_bypass_rgb;
    assign post_i_cfg_we = i_cfg_we;
    assign post_i_cfg_addr = i_cfg_addr;
    assign post_i_cfg_wdata = i_cfg_wdata;
    assign post_i_lut_we = i_load_we && (i_load_space == 3'd4);
    assign post_i_lut_addr = i_load_addr[10:0];
    assign post_i_lut_data = i_load_data[15:0];

    assign o_ready = capture_o_ready;
    assign o_busy = !capture_o_ready || npu_busy_sync_q || post_busy_sync_q;
    assign o_frame_error = capture_o_frame_error;
    assign o_npu_done = npu_o_done;
    assign o_cfg_rdata = post_o_cfg_rdata;
    assign o_vsync = post_o_vsync;
    assign o_hsync = post_o_hsync;
    assign o_rgb = post_o_rgb;
    assign o_box_valid = post_o_box_valid;
    assign o_box_x1 = post_o_box_x1;
    assign o_box_y1 = post_o_box_y1;
    assign o_box_x2 = post_o_box_x2;
    assign o_box_y2 = post_o_box_y2;
    assign o_box_score = post_o_box_score;
    assign o_boxes_done = post_o_boxes_done;
    assign o_result_valid = npu_o_result_valid;
    assign o_result_index = npu_o_result_index;
    assign o_result_data = npu_o_result_data;

    dvp_capture #(
        .p_sensor_width(p_sensor_width),
        .p_sensor_height(p_sensor_height),
        .p_input_type(p_input_type)
    ) u_capture (
        .clk(clk_pixel),
        .rst_n(rst_pixel_n),
        .i_vsync(capture_i_vsync),
        .i_hsync(capture_i_hsync),
        .i_data(capture_i_data),
        .i_inference_done(capture_i_inference_done),
        .o_pixel_we(capture_o_pixel_we),
        .o_pixel_addr(capture_o_pixel_addr),
        .o_pixel_data(capture_o_pixel_data),
        .o_frame_start(capture_o_frame_start),
        .o_ready(capture_o_ready),
        .o_frame_error(capture_o_frame_error),
        .o_bypass_rgb(capture_o_bypass_rgb)
    );

    npu_core u_npu (
        .clk_pixel(clk_pixel),
        .clk_npu(clk_npu),
        .rst_n(rst_npu_n),
        .i_frame_start(npu_i_frame_start),
        .i_pixel_we(npu_i_pixel_we),
        .i_pixel_addr(npu_i_pixel_addr),
        .i_pixel_data(npu_i_pixel_data),
        .i_load_we(npu_i_load_we),
        .i_load_space(npu_i_load_space),
        .i_load_addr(npu_i_load_addr),
        .i_load_data(npu_i_load_data),
        .o_busy(npu_o_busy),
        .o_done(npu_o_done),
        .o_result_valid(npu_o_result_valid),
        .o_result_index(npu_o_result_index),
        .o_result_data(npu_o_result_data)
    );

    bbox_overlay #(
        .p_sensor_width(p_sensor_width),
        .p_sensor_height(p_sensor_height)
    ) u_post (
        .clk_pixel(clk_pixel),
        .clk_npu(clk_npu),
        .rst_pixel_n(rst_pixel_n),
        .rst_npu_n(rst_npu_n),
        .i_result_valid(post_i_result_valid),
        .i_result_index(post_i_result_index),
        .i_result_data(post_i_result_data),
        .i_result_done(post_i_result_done),
        .i_vsync(post_i_vsync),
        .i_hsync(post_i_hsync),
        .i_rgb(post_i_rgb),
        .i_cfg_we(post_i_cfg_we),
        .i_cfg_addr(post_i_cfg_addr),
        .i_cfg_wdata(post_i_cfg_wdata),
        .o_cfg_rdata(post_o_cfg_rdata),
        .i_lut_we(post_i_lut_we),
        .i_lut_addr(post_i_lut_addr),
        .i_lut_data(post_i_lut_data),
        .o_vsync(post_o_vsync),
        .o_hsync(post_o_hsync),
        .o_rgb(post_o_rgb),
        .o_box_valid(post_o_box_valid),
        .o_box_x1(post_o_box_x1),
        .o_box_y1(post_o_box_y1),
        .o_box_x2(post_o_box_x2),
        .o_box_y2(post_o_box_y2),
        .o_box_score(post_o_box_score),
        .o_boxes_done(post_o_boxes_done),
        .o_busy(post_o_busy)
    );
endmodule
`default_nettype wire
