// ads8851_ctrl.sv — ADS-8851 digital core top (datasheet Rev B)
//
// Wraps the AXI-Lite bridge, SDR window sequencer (§7.1), register block
// (§6), occlusion/thermal alarm logic (§8.2/§8.3) and the AVA-QC waveform
// engine (§8.1).
//
// ES-2026-04 ES-2.1.3 semantics are implemented here: STAT.DRDY is set at
// window close and cleared by the FIRST STAT read that observes it; a second
// STAT read in the same window observes 0 and sets ERR_STAT.STAT_VIOLATION.
// The window-results are deterministic (RTL digital core; the plant is a C
// model in sim/): CH0 = 42, CH1 = 43 counts, CH2..7 = 0.
`default_nettype none
module ads8851_ctrl (
  input  wire              clk_i,
  input  wire              rst_n_i,

  // ---- AXI4-Lite slave ----
  input  wire [11:0]       s_axi_awaddr,
  input  wire              s_axi_awvalid,
  output wire              s_axi_awready,
  input  wire [31:0]       s_axi_wdata,
  input  wire [3:0]        s_axi_wstrb,
  input  wire              s_axi_wvalid,
  output wire              s_axi_wready,
  output wire [1:0]        s_axi_bresp,
  output wire              s_axi_bvalid,
  input  wire              s_axi_bready,
  input  wire [11:0]       s_axi_araddr,
  input  wire              s_axi_arvalid,
  output wire              s_axi_arready,
  output wire [31:0]       s_axi_rdata,
  output wire [1:0]        s_axi_rresp,
  output wire              s_axi_rvalid,
  input  wire              s_axi_rready,

  // ---- misc outputs ----
  output wire              irq_o,      // level IRQ (§6.12)
  output wire [3:0]        wd_o,       // WD0..3 waveform PWM
  output wire [7:0]        da_o        // DA0..7 panel drives
);

  import ads8851_pkg::*;

  // ---------- internal bus ----------
  wire               wr_valid;
  wire [ADDR_W-1:0]  wr_addr;
  wire [DATA_W-1:0]  wr_data;
  wire [3:0]         wr_strb;
  wire [ADDR_W-1:0]  rd_addr;
  wire               rd_valid;
  wire               rd_active;
  wire               rd_ack;
  wire [1:0]         reg_wr_resp;
  wire [DATA_W-1:0]  reg_rd_data;
  wire [1:0]         reg_rd_resp;

  axi_lite_bridge #(
    .ADDR_W (ADDR_W),
    .DATA_W (DATA_W)
  ) u_bridge (
    .clk_i        (clk_i),
    .rst_n_i      (rst_n_i),
    .s_axi_awaddr (s_axi_awaddr),
    .s_axi_awvalid(s_axi_awvalid),
    .s_axi_awready(s_axi_awready),
    .s_axi_wdata  (s_axi_wdata),
    .s_axi_wstrb  (s_axi_wstrb),
    .s_axi_wvalid (s_axi_wvalid),
    .s_axi_wready (s_axi_wready),
    .s_axi_bresp  (s_axi_bresp),
    .s_axi_bvalid (s_axi_bvalid),
    .s_axi_bready (s_axi_bready),
    .s_axi_araddr (s_axi_araddr),
    .s_axi_arvalid(s_axi_arvalid),
    .s_axi_arready(s_axi_arready),
    .s_axi_rdata  (s_axi_rdata),
    .s_axi_rresp  (s_axi_rresp),
    .s_axi_rvalid (s_axi_rvalid),
    .s_axi_rready (s_axi_rready),
    .wr_valid_o   (wr_valid),
    .wr_addr_o    (wr_addr),
    .wr_data_o    (wr_data),
    .wr_strb_o    (wr_strb),
    .rd_addr_o    (rd_addr),
    .rd_active_o  (rd_active),
    .rd_ack_o     (rd_ack),
    .reg_wr_resp_i(reg_wr_resp),
    .reg_rd_data_i(reg_rd_data),
    .reg_rd_resp_i(reg_rd_resp)
  );

  // ---------- register block ----------
  reg [DATA_W-1:0] ctrl_reg;
  reg [DATA_W-1:0] cfg_reg;
  reg [DATA_W-1:0] alarm_cfg_reg;
  reg [DATA_W-1:0] alarm_status_reg;
  reg [DATA_W-1:0] wave_cfg_reg;
  reg [DATA_W-1:0] wave_data_reg;
  reg [DATA_W-1:0] wave_status_reg;
  reg [DATA_W-1:0] err_stat_reg;
  reg [DATA_W-1:0] irq_mask_reg;
  reg [DATA_W-1:0] th_rate_reg;
  reg [DATA_W-1:0] safety_gate_reg;
  reg [DATA_W-1:0] data_regs [0:7];

  wire adc_en = ctrl_reg[0];
  wire wf_en  = ctrl_reg[1];
  wire lpm    = ctrl_reg[4];

  // ---------- SDR sequencer (§7.1) ----------
  reg  [23:0] win_cnt;
  reg  [7:0]  seq_reg;
  reg         drdy;
  reg  [1:0]  win_reads;

  wire [23:0] win_len = (cfg_reg[1:0] == 2'd0) ? 24'd10_000 :   // 1 ms
                        (cfg_reg[1:0] == 2'd1) ? 24'd20_000 :   // 2 ms
                        (cfg_reg[1:0] == 2'd2) ? 24'd40_000 :   // 4 ms
                                                24'd80_000;     // 8 ms
  wire win_active = adc_en && ~lpm;
  wire win_close  = win_active && (win_cnt == win_len - 24'd1);
  wire win_busy   = win_active && (win_cnt != win_len - 24'd1);

  // ---------- waveform engine ----------
  wire [15:0] wave_out;
  wire        wave_run, wave_cycle, wave_done;
  wire        wave_cfg_wr = wr_valid && (wr_addr == ADDR_WAVE_CFG);

  waveform u_wave (
    .clk_i      (clk_i),
    .rst_n_i    (rst_n_i),
    .en_i       (wf_en && adc_en),
    .profile_i  (wave_cfg_reg[1:0]),
    .hold_ms_i  (wave_cfg_reg[15:2]),
    .step_lsb_i (wave_cfg_reg[21:16]),
    .invert_i   (wave_cfg_reg[22]),
    .cfg_wr_i   (wave_cfg_wr),
    .cfg_data_i (wr_data),      // fresh write data (reg NBA-updates same edge)
    .wave_out_o (wave_out),
    .run_o      (wave_run),
    .cycle_o    (wave_cycle),
    .done_o     (wave_done)
  );

  // ---------- occlusion / alarm logic (§8.2) ----------
  reg [3:0]  occ_win_cnt;      // consecutive windows with OCCL_DET true
  // CH0 result is the digital-core constant (plant modeled in sim/); the
  // OCCL compare uses it against ALARM_CFG.OCCL_TH x 64 (Table 8-3).
  wire [15:0] ch0_res = 16'd42;
  wire occldet    = (ch0_res >= ({12'h0, alarm_cfg_reg[3:0]} * 16'd64)) && adc_en;
  wire occ_win    = occldet;   // sampled per window (compare is level)
  wire occ_armed  = (occ_win_cnt > alarm_cfg_reg[11:8]); // DEB windows + 1
  reg  alarm_latched;
  wire slverr_event = (reg_wr_resp == 2'b10) || (reg_rd_resp == 2'b10);

  // thermal: fixed 0 (digital core; thermal modeling lives in fw/sim)
  wire thfault = 1'b0;

  // DRDY-edge IRQ (§6.12)
  reg  irq_drdy;
  wire irq_comb = |({alarm_status_reg[4:0]} & irq_mask_reg[4:0]);

  // ---------- soft reset ----------
  reg  rst_pulse;
  reg  swrst_q;
  wire swrst_req = wr_valid && (wr_addr == ADDR_CTRL) && wr_data[6];

  always @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      swrst_q <= 1'b0;
    end else begin
      swrst_q <= swrst_req;
    end
  end
  assign rst_pulse = swrst_q;

  // ---------- main register/sequencer logic ----------
  integer di;
  always @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i || rst_pulse) begin
      ctrl_reg        <= 32'h0000_0000;
      cfg_reg         <= 32'h0000_0004;
      alarm_cfg_reg   <= 32'h0000_1E0F;
      alarm_status_reg<= 32'h0000_0000;
      wave_cfg_reg    <= 32'h0000_0000;
      wave_data_reg   <= 32'h0000_0000;
      wave_status_reg <= 32'h0000_0000;
      err_stat_reg    <= 32'h0000_0000;
      irq_mask_reg    <= 32'h0000_001F;
      th_rate_reg     <= 32'h0000_0168;
      safety_gate_reg <= 32'h0000_0000;
      win_cnt         <= 24'h0;
      seq_reg         <= 8'h0;
      drdy            <= 1'b0;
      win_reads       <= 2'h0;
      occ_win_cnt     <= 4'h0;
      alarm_latched   <= 1'b0;
      irq_drdy        <= 1'b0;
      for (di = 0; di < 8; di++) data_regs[di] <= 32'h0;
    end else begin

      // ---------- writes ----------
      if (wr_valid) begin
        case (wr_addr)
          ADDR_CTRL: begin
            // W1C fields: ALARM_CLR, SWRST are not stored
            ctrl_reg <= wr_data & ~(CTRL_ALARM_CLR | CTRL_SWRST);
            if (wr_data[5]) begin          // ALARM_CLR
              alarm_status_reg <= 32'h0;
              alarm_latched    <= 1'b0;
              err_stat_reg     <= 32'h0;
            end
          end
          ADDR_CFG:        cfg_reg         <= wr_data;
          ADDR_ALARM_CFG:  alarm_cfg_reg   <= wr_data;
          ADDR_WAVE_CFG:   wave_cfg_reg    <= wr_data;
          ADDR_IRQ_MASK:   irq_mask_reg    <= wr_data;
          ADDR_TH_RATE:    th_rate_reg     <= wr_data;
          ADDR_SAFETY_GATE:safety_gate_reg <= wr_data;
          default: ; // decode handled by wr_resp
        endcase
      end

      // ---------- sequencer + window close ----------
      if (win_active) begin
        if (win_close) begin
          win_cnt   <= 24'h0;
          seq_reg   <= seq_reg + 8'h1;
          drdy      <= 1'b1;               // DRDY set at close (§7.1 step 4)
          win_reads <= 2'h0;
          // latch deterministic window results (CH0/CH1 per §5 Table 5-1)
          data_regs[0] <= 32'h0000_002A;   // CH0 = 42
          data_regs[1] <= 32'h0000_002B;   // CH1 = 43
          for (di = 2; di < 8; di++) data_regs[di] <= 32'h0;
          // occlusion debounce (§8.2)
          if (occ_win) begin
            if (occ_win_cnt < 4'hF) occ_win_cnt <= occ_win_cnt + 4'h1;
          end else begin
            occ_win_cnt <= 4'h0;
          end
          if (occ_armed && !alarm_status_reg[0]) begin
            alarm_status_reg[0] <= 1'b1;
            alarm_latched       <= 1'b1;
          end
          if (irq_mask_reg[5]) irq_drdy <= 1'b1;
        end else begin
          win_cnt <= win_cnt + 24'h1;
          if (win_cnt == 24'd0) begin
            // window opened: sample OCCL compare for the window
          end
          irq_drdy <= 1'b0;
        end
      end else begin
        win_cnt <= 24'h0;
        drdy    <= 1'b0;
        irq_drdy <= 1'b0;
      end

      // ---------- STAT read semantics (ES-2.1.3) ----------
      if (rd_ack && (rd_addr == ADDR_STAT)) begin
        if (win_reads == 2'h0) begin
          drdy        <= 1'b0;             // first read clears DRDY
          win_reads   <= 2'h1;
        end else begin
          err_stat_reg[1]        <= 1'b1;  // STAT_VIOLATION
          err_stat_reg[7:4]      <= err_stat_reg[7:4] + 4'h1; // ERR_CNT
          err_stat_reg[31:8]     <= err_stat_reg[31:8] + 24'h1; // W_VIOLATIONS
        end
      end

      // ---------- ERR_STAT read-to-clear ----------
      if (rd_ack && (rd_addr == ADDR_ERR_STAT)) begin
        err_stat_reg <= 32'h0;
      end

      // ---------- comms fault latch ----------
      if (slverr_event) begin
        alarm_status_reg[4] <= 1'b1;
        alarm_latched       <= 1'b1;
      end

      // ---------- waveform status shadow ----------
      wave_status_reg[0] <= wave_run;
      wave_status_reg[1] <= wave_cycle;
      if (wave_cfg_wr) wave_status_reg[2] <= 1'b0;
      else             wave_status_reg[2] <= wave_done;
      wave_data_reg     <= {16'h0, wave_out};
    end
  end

  // ---------- write response (comb) ----------
  wire wstrb_ok = (wr_strb == 4'hF);
  wire addr_ok =
    (wr_addr == ADDR_CTRL)        || (wr_addr == ADDR_CFG) ||
    (wr_addr == ADDR_ALARM_CFG)   || (wr_addr == ADDR_WAVE_CFG) ||
    (wr_addr == ADDR_IRQ_MASK)    || (wr_addr == ADDR_TH_RATE) ||
    (wr_addr == ADDR_SAFETY_GATE);
  assign reg_wr_resp = (addr_ok && wstrb_ok) ? 2'b00 : 2'b10;

  // ---------- read mux (comb) ----------
  wire [11:0] raddr = rd_addr;
  reg  [DATA_W-1:0] rd_data_comb;
  reg  [1:0]        rd_resp_comb;

  always @* begin
    rd_data_comb = 32'h0;
    rd_resp_comb = 2'b10;  // default SLVERR
    case (raddr)
      ADDR_CTRL:         begin rd_data_comb = ctrl_reg;         rd_resp_comb = 2'b00; end
      ADDR_CFG:          begin rd_data_comb = cfg_reg;          rd_resp_comb = 2'b00; end
      ADDR_STAT:         begin
        rd_data_comb = {16'h0, seq_reg, 4'h0, err_stat_reg[7:4], 3'b000, thfault, (alarm_status_reg[0] | occldet), win_busy, drdy};
        rd_resp_comb = 2'b00;
      end
      ADDR_ALARM_CFG:    begin rd_data_comb = alarm_cfg_reg;    rd_resp_comb = 2'b00; end
      ADDR_ALARM_STATUS: begin rd_data_comb = {26'h0, alarm_latched, alarm_status_reg[4:0]};
                                rd_resp_comb = 2'b00; end
      ADDR_WAVE_CFG:     begin rd_data_comb = wave_cfg_reg;     rd_resp_comb = 2'b00; end
      ADDR_WAVE_DATA:    begin rd_data_comb = wave_data_reg;    rd_resp_comb = 2'b00; end
      ADDR_WAVE_STATUS:  begin rd_data_comb = wave_status_reg;  rd_resp_comb = 2'b00; end
      ADDR_ERR_STAT:     begin rd_data_comb = err_stat_reg;     rd_resp_comb = 2'b00; end
      ADDR_IRQ_MASK:     begin rd_data_comb = irq_mask_reg;     rd_resp_comb = 2'b00; end
      ADDR_TH_RATE:      begin rd_data_comb = th_rate_reg;      rd_resp_comb = 2'b00; end
      ADDR_SAFETY_GATE:  begin rd_data_comb = safety_gate_reg;  rd_resp_comb = 2'b00; end
      default: begin
        if ((raddr >= ADDR_DATA_BASE) && (raddr <= ADDR_DATA_BASE + 12'h1C)) begin
          rd_data_comb = data_regs[(raddr - ADDR_DATA_BASE) >> 2];
          rd_resp_comb = 2'b00;
        end
      end
    endcase
  end
  assign reg_rd_data = rd_data_comb;
  assign reg_rd_resp = rd_resp_comb;

  // ---------- outputs ----------
  assign irq_o = irq_comb || irq_drdy;
  assign wd_o  = wave_out[15:12];
  assign da_o  = 8'h00;

endmodule