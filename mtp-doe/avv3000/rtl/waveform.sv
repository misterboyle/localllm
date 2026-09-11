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
//
// cfg_wr_i pulses on a WAVE_CFG write; cfg_data_i is the write data. The
// register file updates its mirror with NBA on the SAME edge, so the engine
// must decode NEW config from cfg_data_i inside the cfg_wr branch — the
// profile_i/hold_ms_i/... ports lag by one cycle and are stale there.
`default_nettype none
module waveform #(
  parameter int ACLK_MHZ = 10
)(
  input  wire               clk_i,
  input  wire               rst_n_i,
  input  wire               en_i,          // WF_EN && ADC_EN
  input  wire [1:0]         profile_i,     // WAVE_CFG[1:0]
  input  wire [13:0]        hold_ms_i,     // WAVE_CFG[15:2] (1..16383 ms).
                                           // 14-bit keeps hold_lat *
                                           // CLKS_PER_MS inside tick_cnt
  input  wire [5:0]         step_lsb_i,    // WAVE_CFG[21:16]
  input  wire               invert_i,      // WAVE_CFG[22]
  input  wire               cfg_wr_i,      // any WAVE_CFG write
  input  wire [31:0]        cfg_data_i,    // WAVE_CFG write data (with cfg_wr_i)
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

  reg [13:0] hold_lat;      // applied cadence (latched at cycle start)
  reg [5:0]  step_lat;      // applied step
  reg [15:0] wave_q;
  reg [27:0] tick_cnt;
  reg [4:0]  step_cnt;
  reg [7:0]  up_steps, dn_steps;
  reg [10:0] holdtop_ms;    // profile-local top dwell, ms (app note §3: B =
                            // 1200). 11-bit width bounds holdtop_ms *
                            // CLKS_PER_MS inside tick_cnt; an 8-bit store
                            // here truncates profile B's dwell to 176 ms
  reg        sense;         // 0 = ascending, 1 = descending
  reg        iter_reg;      // 1 once a full cycle completed
  reg [1:0]  profile_applied; // profile the running engine was armed with

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

  // config decode for the cfg_wr edge (fresh write data, NOT the stale
  // register ports — see header note)
  wire [1:0]  cfg_prof = cfg_data_i[1:0];
  wire [13:0] cfg_hold = cfg_data_i[15:2];
  wire [5:0]  cfg_step = cfg_data_i[21:16];
  wire        cfg_inv  = cfg_data_i[22];

  always @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
      st        <= ST_IDLE;
      wave_q    <= 16'h0000;
      hold_lat  <= 14'h0000;
      step_lat  <= 6'h00;
      tick_cnt  <= 28'h0;
      step_cnt  <= 5'h0;
      sense     <= 1'b0;
      iter_reg  <= 1'b0;
      profile_applied <= 2'd0;
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
            profile_applied <= profile_i;
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
              hold_lat <= hold_ms_i;        // deferred HOLD_MS applies here
                                            // (ES-2.1.2: next cycle start)
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

      // ---- config write handling (decode cfg_data_i, see header) ----
      if (cfg_wr_i) begin
        done_o <= 1'b0;
        if (cfg_prof == 2'd0) begin
          // stop
          st              <= ST_IDLE;
          run_o           <= 1'b0;
          iter_reg        <= 1'b0;
          step_cnt        <= 5'h0;
          profile_applied <= 2'd0;
        end else if ((st == ST_IDLE || st == ST_DONE_ST) && en_i) begin
          // (re)start with the written params; wave_q is NOT cleared
          // (profile switches re-arm the ramp from the current output).
          // en_i-gated per §6.15 (RUN = WF_EN & PROFILE != 0); with the
          // engine disabled the config lands in the register and the
          // ST_IDLE start path picks it up when en_i rises.
          hold_lat <= cfg_hold;
          step_lat <= cfg_step;
          tick_cnt <= 28'h0;
          sense    <= cfg_inv;
          st       <= ST_RAMP_UP;
          run_o    <= 1'b1;
          profile_applied <= cfg_prof;
        end else begin
          // running: PROFILE/STEP/INVERT apply immediately; a held HOLD_MS
          // change applies at the next cycle start (ES-2.1.2)
          step_lat <= cfg_step;
          sense    <= cfg_inv;
          if (cfg_prof != profile_applied) begin
            // profile switch mid-run: re-arm the ramp from the current
            // output (geometry and stepper restart)
            step_cnt <= 5'h0;
          end
          profile_applied <= cfg_prof;
        end
      end
    end
  end

endmodule