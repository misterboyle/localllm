// waveform.sv — AVA-QC staircase waveform engine (datasheet §8.1 / app note)
//
// Golden implementation of the AVA-QC profiles (appnote-ava-qc-staircase.md
// §3). Seeded defect F1 (feature): the engine is a stub — run_o stays 0 and
// wave_out_o stays 0 — the feature has to be implemented by the agent.
//
// Profile tables (app note §3 — authoritative):
//   A (1): up 10 steps, down 10 steps, loop          (hold_ms default 500)
//   B (2): up 8, hold top 1200 ms, down 8, loop      (hold_ms default 800)
//   C (3): up 14 steps, one-way, DONE sticky         (hold_ms default 200)
// Cadence: one step per hold_ms x 1 ms (ACLK 10 MHz).
// ES-2026-04 ES-2.1.2: HOLD_MS changes take effect at the NEXT cycle start;
//   PROFILE / STEP_LSB / INVERT apply immediately.
`default_nettype none
module waveform #(
  parameter int ACLK_MHZ = 10
)(
  input  wire               clk_i,
  input  wire               rst_n_i,
  input  wire               en_i,          // WF_EN && ADC_EN
  input  wire [1:0]         profile_i,     // WAVE_CFG[1:0]
  input  wire [7:0]         hold_ms_i,     // WAVE_CFG[9:2]
  input  wire [5:0]         step_lsb_i,    // WAVE_CFG[15:10]
  input  wire               invert_i,      // WAVE_CFG[16]
  input  wire               cfg_wr_i,      // any WAVE_CFG write
  output reg  [15:0]        wave_out_o,    // WAVE_DATA
  output reg                run_o,         // WAVE_STATUS.RUN
  output wire               cycle_o,       // WAVE_STATUS.CYCLE (iter >= 2)
  output reg                done_o         // WAVE_STATUS.DONE
);

  // profile constants (app note §3)
  localparam int UP_A = 10, DN_A = 10, HOLDTOP_A = 0;
  localparam int UP_B = 8,  DN_B = 8,  HOLDTOP_B = 1200;
  localparam int UP_C = 14, DN_C = 0;

  localparam int CLKS_PER_MS = ACLK_MHZ * 1000;   // 10_000 @ 10 MHz

  localparam logic [1:0] PROF_A = 2'd1;
  localparam logic [1:0] PROF_B = 2'd2;
  localparam logic [1:0] PROF_C = 2'd3;

  typedef enum logic [2:0] {
    ST_IDLE, ST_RAMP_UP, ST_HOLD_TOP, ST_RAMP_DN, ST_DONE_ST
  } state_t;

  state_t st;

  reg [7:0]  hold_lat;      // applied cadence (latched at cycle start)
  reg [5:0]  step_lat;      // applied step
  reg [15:0] wave_q;
  reg [27:0] tick_cnt;
  reg [4:0]  step_cnt;
  reg [7:0]  up_steps, dn_steps, holdtop_ms;
  reg        sense;         // 0 = ascending, 1 = descending
  reg        iter_reg;      // 1 once a full cycle completed
  reg [1:0]  profile_prev;  // profile seen last cycle (mid-run change detect)

  assign wave_out_o = wave_q;
  assign cycle_o    = iter_reg;   // 1 during cycle 2..N (§6.15)

  function automatic logic running(state_t s);
    return (s != ST_IDLE && s != ST_DONE_ST);
  endfunction

  // load profile geometry
  always @* begin
    case (profile_i)
      PROF_A: begin up_steps = UP_A; dn_steps = DN_A; holdtop_ms = HOLDTOP_A; end
      PROF_B: begin up_steps = UP_B; dn_steps = DN_B; holdtop_ms = HOLDTOP_B; end
      PROF_C: begin up_steps = UP_C; dn_steps = DN_C; holdtop_ms = 0;         end
      default: begin up_steps = 0;   dn_steps = 0;   holdtop_ms = 0;         end
    endcase
  end

  wire tick = (tick_cnt >= (hold_lat * CLKS_PER_MS) - 1);

  always @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      st        <= ST_IDLE;
      wave_q    <= 16'h0000;
      hold_lat  <= 8'h00;
      step_lat  <= 6'h00;
      tick_cnt  <= 28'h0;
      step_cnt  <= 5'h0;
      sense     <= 1'b0;
      iter_reg  <= 1'b0;
      profile_prev <= 2'd0;
      run_o     <= 1'b0;
      done_o    <= 1'b0;
    end else begin
      done_o <= 1'b0;   // default: pulsing DONE

      case (st)
        ST_IDLE: begin
          if (en_i && (profile_i != 2'd0)) begin
            hold_lat <= hold_ms_i;
            step_lat <= step_lsb_i;
            sense    <= invert_i;
            st       <= ST_RAMP_UP;
            run_o    <= 1'b1;
          end
        end

        ST_RAMP_UP: begin
          if (tick) begin
            tick_cnt <= 28'h0;
            step_cnt <= step_cnt + 1'b1;
            if (sense) begin
              if (wave_q >= step_lat) wave_q <= wave_q - {{10{1'b0}}, step_lat};
              else                    wave_q <= 16'h0000;
            end else begin
              if (wave_q < 16'hFFFF - {{10{1'b0}}, step_lat})
                wave_q <= wave_q + {{10{1'b0}}, step_lat};
              else
                wave_q <= 16'hFFFF;
            end
            if (step_cnt + 1'b1 >= up_steps) begin
              if (holdtop_ms != 0) begin
                st       <= ST_HOLD_TOP;
              end else if (dn_steps != 0) begin
                sense    <= ~sense;
                step_cnt <= 5'h0;
                st       <= ST_RAMP_DN;
              end else begin
                // one-way profile (C): done sticky, stop
                done_o <= 1'b1;
                st     <= ST_DONE_ST;
                run_o  <= 1'b0;
              end
            end
          end else begin
            tick_cnt <= tick_cnt + 1'b1;
          end
        end

        ST_HOLD_TOP: begin
          if (tick_cnt >= (holdtop_ms * CLKS_PER_MS) - 1) begin
            tick_cnt <= 28'h0;
            sense    <= ~sense;
            step_cnt <= 5'h0;
            st       <= ST_RAMP_DN;
          end else begin
            tick_cnt <= tick_cnt + 1'b1;
          end
        end

        ST_RAMP_DN: begin
          if (tick) begin
            tick_cnt <= 28'h0;
            step_cnt <= step_cnt + 1'b1;
            if (sense) begin
              if (wave_q >= step_lat) wave_q <= wave_q - {{10{1'b0}}, step_lat};
              else                    wave_q <= 16'h0000;
            end else begin
              wave_q <= wave_q + {{10{1'b0}}, step_lat};
            end
            if (step_cnt + 1'b1 >= dn_steps) begin
              // cycle end
              done_o   <= 1'b1;             // pulse
              iter_reg <= 1'b1;
              sense    <= invert_i;         // re-arm
              step_cnt <= 5'h0;
              st       <= ST_RAMP_UP;       // loop profiles only reach here
            end
          end else begin
            tick_cnt <= tick_cnt + 1'b1;
          end
        end

        ST_DONE_ST: begin
          done_o <= 1'b1;                   // sticky for one-way profiles
          run_o  <= 1'b0;
        end

      endcase

      // ---- config write handling ----
      if (cfg_wr_i) begin
        done_o <= 1'b0;
        if (profile_i == 2'd0) begin
          // stop
          st     <= ST_IDLE;
          run_o  <= 1'b0;
          iter_reg <= 1'b0;
          step_cnt <= 5'h0;
        end else begin
          if (st == ST_IDLE || st == ST_DONE_ST) begin
            // (re)start with latched params
            hold_lat <= hold_ms_i;
            step_lat <= step_lsb_i;
            sense    <= invert_i;
            st       <= ST_RAMP_UP;
            run_o    <= 1'b1;
          end else begin
            // running: STEP/PROFILE apply now, HOLD_MS at next cycle start (ES-2.1.2)
            step_lat <= step_lsb_i;    // applied immediately
            if (profile_i != profile_prev) begin
              // profile switch mid-run: re-arm the ramp from the current
              // output (geometry and stepper restart)
              step_cnt <= 5'h0;
              sense    <= invert_i;
            end
          end
        end
      end
      // track the last-seen profile for mid-run change detection
      profile_prev <= profile_i;
    end
  end

endmodule