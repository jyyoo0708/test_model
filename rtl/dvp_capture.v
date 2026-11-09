`default_nettype none
module dvp_capture #(
    parameter integer p_sensor_width = 96,
    parameter integer p_sensor_height = 96,
    parameter integer p_input_type = 0
) (
    input wire clk,
    input wire rst_n,
    input wire i_vsync,
    input wire i_hsync,
    input wire [23:0] i_data,
    input wire i_inference_done,
    output reg o_pixel_we,
    output reg [13:0] o_pixel_addr,
    output reg signed [7:0] o_pixel_data,
    output reg o_frame_start,
    output wire o_ready,
    output reg o_frame_error,
    output wire [23:0] o_bypass_rgb
);
    localparam integer p_max_dim = (p_sensor_width > p_sensor_height) ? p_sensor_width : p_sensor_height;
    localparam integer p_resize_w_raw = (p_sensor_width*96 + p_max_dim/2)/p_max_dim;
    localparam integer p_resize_w = (p_resize_w_raw<1) ? 1 : p_resize_w_raw;
    localparam integer p_resize_h_raw = (p_sensor_height*96 + p_max_dim/2)/p_max_dim;
    localparam integer p_resize_h = (p_resize_h_raw<1) ? 1 : p_resize_h_raw;
    localparam integer p_left = (96-p_resize_w)/2;
    localparam integer p_top = (96-p_resize_h)/2;
    localparam [1:0] p_prep=2'd0, p_ready=2'd1, p_capture=2'd2, p_wait=2'd3;
    reg [1:0] state_q;
    reg vsync_q, hsync_q;
    reg [13:0] prep_q;
    reg [31:0] x_q,y_q,dx_q,dy_q;
    reg [13:0] count_q;
    reg bad_q;
    wire [7:0] red, green, blue;
    wire [15:0] unused_luma_sum;
    wire [7:0] gray;
    wire [15:0] quant_product;
    wire [15:0] unused_quant_quotient;
    wire [7:0] quant_gray;
    wire [31:0] unused_dest_addr;
    localparam [31:0] p_expected_count = p_resize_w*p_resize_h;
    wire [31:0] sample_x, sample_y;
    assign red = (p_input_type==0) ? i_data[7:0] :
                 (p_input_type==1) ? i_data[23:16] : {i_data[15:11],i_data[15:13]};
    assign green = (p_input_type==0) ? i_data[7:0] :
                   (p_input_type==1) ? i_data[15:8] : {i_data[10:5],i_data[10:9]};
    assign blue = (p_input_type==0) ? i_data[7:0] :
                  (p_input_type==1) ? i_data[7:0] : {i_data[4:0],i_data[4:2]};
    assign o_bypass_rgb = {red,green,blue};
    assign unused_luma_sum = (16'd77*red)+(16'd150*green)+(16'd29*blue)+16'd128;
    assign gray = (p_input_type==0) ? i_data[7:0] : unused_luma_sum[15:8];
    assign quant_product = (16'd127*gray)+16'd127;
    assign unused_quant_quotient = quant_product/16'd255;
    assign quant_gray = unused_quant_quotient[7:0];
    assign unused_dest_addr = ((dy_q+p_top)*96)+dx_q+p_left;
    assign sample_x = (dx_q*p_sensor_width)/p_resize_w;
    assign sample_y = (dy_q*p_sensor_height)/p_resize_h;
    assign o_ready = (state_q==p_ready);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q<=p_prep; prep_q<=14'd0;
            vsync_q<=1'b0; hsync_q<=1'b0;
            x_q<=32'd0; y_q<=32'd0; dx_q<=32'd0; dy_q<=32'd0;
            count_q<=14'd0; bad_q<=1'b0;
            o_pixel_we<=1'b0; o_pixel_addr<=14'd0; o_pixel_data<=8'sd0;
            o_frame_start<=1'b0; o_frame_error<=1'b0;
        end else begin
            vsync_q<=i_vsync; hsync_q<=i_hsync;
            o_pixel_we<=1'b0; o_frame_start<=1'b0; o_frame_error<=1'b0;
            case (state_q)
                p_prep: begin
                    o_pixel_we<=1'b1; o_pixel_addr<=prep_q; o_pixel_data<=8'sd57;
                    if (prep_q==14'd9215) begin
                        state_q<=p_ready; prep_q<=14'd0;
                    end else begin
                        prep_q<=prep_q+14'd1;
                    end
                end
                p_ready: begin
                    if (i_vsync && !vsync_q) begin
                        state_q<=p_capture;
                        x_q<=32'd0; y_q<=32'd0; dx_q<=32'd0; dy_q<=32'd0;
                        count_q<=14'd0; bad_q<=i_hsync;
                    end
                end
                p_capture: begin
                    if (!i_vsync) begin
                        if (!bad_q && ({18'd0,count_q}==p_expected_count) &&
                            (y_q==p_sensor_height) && !hsync_q) begin
                            o_frame_start<=1'b1; state_q<=p_wait;
                        end else begin
                            o_frame_error<=1'b1; state_q<=p_prep;
                        end
                    end else if (i_hsync) begin
                        if ((x_q>=p_sensor_width) || (y_q>=p_sensor_height)) begin
                            bad_q<=1'b1;
                        end
                        if ((dy_q<p_resize_h) && (dx_q<p_resize_w) &&
                            (x_q==sample_x) && (y_q==sample_y)) begin
                            o_pixel_we<=1'b1;
                            o_pixel_addr<=unused_dest_addr[13:0];
                            o_pixel_data<=$signed(quant_gray);
                            count_q<=count_q+14'd1;
                            if (dx_q==(p_resize_w-1)) begin
                                dx_q<=32'd0; dy_q<=dy_q+32'd1;
                            end else begin
                                dx_q<=dx_q+32'd1;
                            end
                        end
                        x_q<=x_q+32'd1;
                    end else if (hsync_q) begin
                        if (x_q!=p_sensor_width) begin bad_q<=1'b1; end
                        x_q<=32'd0; y_q<=y_q+32'd1;
                    end
                end
                p_wait: begin
                    if (i_inference_done) begin state_q<=p_prep; end
                end
                default: begin state_q<=p_prep; end
            endcase
        end
    end
endmodule
`default_nettype wire
