// axi_lite_bridge.sv — AXI4-Lite slave bridge (AM004 ch.4 compatible)
//
// Converts the 5-channel AXI4-Lite port into a simple internal
// request/response handshake for the ADS-8851 register block.
//
// Golden behavior (AM004 §4.3 — THE invariant): every committed write
// produces exactly one B response, in order. A new AW/W arriving while a
// previous write's B is still pending must NOT suppress or drop the pending
// response. This is the invariant broken by seeded defect B4.
//
// Seeded B4 (rtl): the B path uses a SINGLE outstanding-response slot and
// overwrites it when a new write commits before the previous B is accepted —
// the first write's B is dropped. See seed/bugs/.
`default_nettype none
module axi_lite_bridge #(
  parameter int ADDR_W = 12,
  parameter int DATA_W = 32
)(
  input  wire                        clk_i,
  input  wire                        rst_n_i,

  // ---- AXI4-Lite slave ----
  input  wire [ADDR_W-1:0]           s_axi_awaddr,
  input  wire                        s_axi_awvalid,
  output wire                        s_axi_awready,
  input  wire [DATA_W-1:0]           s_axi_wdata,
  input  wire [DATA_W/8-1:0]         s_axi_wstrb,
  input  wire                        s_axi_wvalid,
  output wire                        s_axi_wready,
  output wire [1:0]                  s_axi_bresp,
  output wire                        s_axi_bvalid,
  input  wire                        s_axi_bready,
  input  wire [ADDR_W-1:0]           s_axi_araddr,
  input  wire                        s_axi_arvalid,
  output wire                        s_axi_arready,
  output wire [DATA_W-1:0]           s_axi_rdata,
  output wire [1:0]                  s_axi_rresp,
  output wire                        s_axi_rvalid,
  input  wire                        s_axi_rready,

  // ---- internal register-block interface ----
  output reg                         wr_valid_o,
  output reg  [ADDR_W-1:0]           wr_addr_o,
  output reg  [DATA_W-1:0]           wr_data_o,
  output reg  [DATA_W/8-1:0]         wr_strb_o,
  output wire [ADDR_W-1:0]           rd_addr_o,
  output wire                        rd_valid_o,   // pulse on AR accept
  output wire                        rd_active_o,  // read in flight
  output wire                        rd_ack_o,     // read accepted by host
  input  wire [1:0]                  reg_wr_resp_i,
  input  wire [DATA_W-1:0]           reg_rd_data_i,
  input  wire [1:0]                  reg_rd_resp_i
);

  // ---------- write address / data capture ----------
  reg  [ADDR_W-1:0]  aw_q;
  reg  [DATA_W-1:0]  w_q;
  reg  [DATA_W/8-1:0] strb_q;
  reg                aw_have, w_have;
  reg                wr_inflight;

  wire wr_done = (wstate == WS_COMMIT);

  assign s_axi_awready = ~aw_have;
  assign s_axi_wready  = ~w_have && ~wr_inflight;

  always @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      aw_have  <= 1'b0;
      w_have   <= 1'b0;
    end else begin
      if (s_axi_awvalid && s_axi_awready) begin
        aw_q    <= s_axi_awaddr;
        aw_have <= 1'b1;
      end
      if (s_axi_wvalid && s_axi_wready) begin
        w_q     <= s_axi_wdata;
        strb_q  <= s_axi_wstrb;
        w_have  <= 1'b1;
      end
      if (wr_done) begin
        aw_have  <= 1'b0;
        w_have   <= 1'b0;
      end
    end
  end

  // ---------- write-commit state machine ----------
  reg [1:0] wstate;
  localparam WS_IDLE   = 2'd0;
  localparam WS_COMMIT = 2'd1;
  localparam WS_BIDLE  = 2'd2;

  reg  [1:0] b_resp_q;

  assign wr_valid_o = (wstate == WS_COMMIT);

  always @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      wstate      <= WS_IDLE;
      wr_addr_o   <= {ADDR_W{1'b0}};
      wr_data_o   <= {DATA_W{1'b0}};
      wr_strb_o   <= {(DATA_W/8){1'b0}};
      b_resp_q    <= 2'b00;
      wr_inflight <= 1'b0;
    end else begin
      case (wstate)
        WS_IDLE: begin
          if (aw_have && w_have) begin
            wr_addr_o  <= aw_q;
            wr_data_o  <= w_q;
            wr_strb_o  <= strb_q;
            wr_inflight <= 1'b1;
            wstate     <= WS_COMMIT;
          end
        end
        WS_COMMIT: begin
          b_resp_q    <= reg_wr_resp_i;   // comb resp from register block
          wr_inflight <= 1'b0;
          wstate      <= WS_BIDLE;
        end
        WS_BIDLE: begin
          wstate <= WS_IDLE;
        end
      endcase
    end
  end

  // ---------- B response path (GOLDEN: 2-deep FIFO, never drops) ----------
  reg        b_fifo_valid [1:0];
  reg [1:0]  b_fifo_resp  [1:0];
  reg        b_fifo_head;
  reg        b_fifo_tail;

  assign s_axi_bvalid = b_fifo_valid[b_fifo_head];
  assign s_axi_bresp  = b_fifo_resp[b_fifo_head];

  wire b_pop  = s_axi_bvalid && s_axi_bready;
  wire b_push = (wstate == WS_BIDLE);   // one push per committed write

  integer i;
  always @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      for (i = 0; i < 2; i++) begin
        b_fifo_valid[i] <= 1'b0;
        b_fifo_resp[i]  <= 2'b00;
      end
      b_fifo_head <= 1'b0;
      b_fifo_tail <= 1'b0;
    end else begin
      if (b_push && ~b_pop) begin
        b_fifo_valid[b_fifo_tail] <= 1'b1;
        b_fifo_resp[b_fifo_tail]  <= b_resp_q;
        b_fifo_tail <= ~b_fifo_tail;
      end else if (~b_push && b_pop) begin
        b_fifo_valid[b_fifo_head] <= 1'b0;
        b_fifo_head <= ~b_fifo_head;
      end else if (b_push && b_pop) begin
        b_fifo_valid[b_fifo_head] <= 1'b1;
        b_fifo_resp[b_fifo_head]  <= b_resp_q;
      end
    end
  end

  // ---------- read path ----------
  reg                r_pend;
  reg  [ADDR_W-1:0]  ar_q;

  assign s_axi_arready = ~r_pend;
  assign s_axi_rvalid  = r_pend;
  assign s_axi_rdata   = reg_rd_data_i;   // comb data for ar_q
  assign s_axi_rresp   = reg_rd_resp_i;

  assign rd_valid_o = s_axi_arvalid && s_axi_arready; // pulse on AR accept
  assign rd_addr_o  = s_axi_araddr;                   // comb: decode in reg block
  assign rd_active_o = r_pend;
  assign rd_ack_o    = r_pend && s_axi_rready;        // host accepted R

  always @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      r_pend <= 1'b0;
      ar_q   <= {ADDR_W{1'b0}};
    end else begin
      if (rd_valid_o) begin
        ar_q   <= s_axi_araddr;
        r_pend <= 1'b1;
      end else if (rd_ack_o) begin
        r_pend <= 1'b0;
      end
    end
  end

endmodule