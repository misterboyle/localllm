// t_waveform.cc — Verilator testbench: AVA-QC staircase engine (F1, §8.1).
// Rung: r_rtl (15 pts, t_waveform half). Fails at start on the seeded tree
// (waveform.sv is a stub: WAVE_DATA stays 0, RUN stays 0).
// Covers: profile A ramp up/down + loop + DONE pulse; profile C one-way +
// sticky DONE + RUN drop; ES-2.1.2 mid-cycle HOLD_MS deferral.
#include "Vads8851_ctrl.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <cstdio>

static Vads8851_ctrl* top;
static vluint64_t cyc = 0;
static int failures = 0;
static VerilatedVcdC* tfp = nullptr;

#define CHECK(cond, msg) do { \
  if (!(cond)) { printf("FAIL line %d: %s\n", __LINE__, msg); failures++; } \
  else { printf("  ok: %s\n", msg); } \
} while (0)

static void tick() {
  if (tfp) tfp->dump((vluint64_t)(cyc * 2));
  top->clk_i = 0; top->eval();
  if (tfp) tfp->dump((vluint64_t)(cyc * 2 + 1));
  top->clk_i = 1; top->eval();
  cyc++;
}

static void reset(int n = 10) {
  top->rst_n_i = 0;
  for (int i = 0; i < n; i++) tick();
  top->rst_n_i = 1;
}

static void axi_idle() {
  top->s_axi_awvalid = 0; top->s_axi_wvalid = 0; top->s_axi_wstrb = 0xF;
  top->s_axi_bready  = 1;
  top->s_axi_arvalid = 0; top->s_axi_rready = 1;
}

static void axi_write(uint32_t addr, uint32_t data) {
  top->s_axi_awaddr  = addr;
  top->s_axi_awvalid = 1;
  top->s_axi_wdata   = data;
  top->s_axi_wstrb   = 0xF;
  top->s_axi_wvalid  = 1;
  int aw = 0, w = 0, b = 0, timeout = 0;
  while (!(aw && w && b)) {
    int awready = top->s_axi_awready && top->s_axi_awvalid && !aw;
    int wready  = top->s_axi_wready  && top->s_axi_wvalid  && !w;
    tick();                                   // transfers clock in at edge
    if (awready) aw = 1;
    if (wready)  w  = 1;
    if (aw && w) { top->s_axi_awvalid = 0; top->s_axi_wvalid = 0; }
    if (top->s_axi_bvalid) b = 1;
    if (++timeout > 60) break;
  }
  axi_idle();
}

static uint32_t axi_read(uint32_t addr) {
  top->s_axi_araddr  = addr;
  top->s_axi_arvalid = 1;
  int timeout = 0;
  while (1) {
    int ready = top->s_axi_arready && top->s_axi_arvalid;
    tick();
    if (ready) { top->s_axi_arvalid = 0; break; }
    if (++timeout > 40) break;
  }
  timeout = 0;
  while (1) {
    if (top->s_axi_rvalid) {
      uint32_t d = top->s_axi_rdata;
      tick();
      return d;
    }
    tick();
    if (++timeout > 40) break;
  }
  return 0xDEADBEEF;
}

static vluint64_t wave_val() { return axi_read(0x18) & 0xFFFF; }

static vluint64_t wave_status() { return axi_read(0x1C); }

// advance until wave_out reaches want (with timeout); returns cycles used
static vluint64_t wait_wave(int want, vluint64_t max_cyc) {
  vluint64_t start = cyc;
  while (cyc - start < max_cyc) {
    if ((int)wave_val() == want) return cyc - start;
  }
  return cyc - start;
}

// ------------------------------------------------------------------
int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Verilated::traceEverOn(true);
  top = new Vads8851_ctrl;
  tfp = new VerilatedVcdC;
  top->trace(tfp, 99);
  tfp->open("trace.vcd");
  top->clk_i = 0;
  reset();
  axi_idle();

  printf("== t_waveform: AVA-QC staircase engine ==\n");

  // ---- setup: ADC on, 1 ms windows, waveform off by default ----
  axi_write(0x00, 0x01);            // ADC_EN
  axi_write(0x04, 0x00);            // 1 ms window (fast test)

  // ---- profile A: 10 up + 10 down @ 12 LSB, hold 1 ms, loop ----
  axi_write(0x14, (1 << 0) | (1 << 2) | (12 << 10));   // PROFILE=A, HOLD_MS=1, STEP=12
  axi_write(0x00, 0x01 | 0x02);     // ADC_EN | WF_EN

  CHECK(wave_val() == 0, "wave starts at 0");
  CHECK((wave_status() & 1) == 1, "RUN asserted (profile A)");

  // up ramp: 12, 24, ..., 120 at 10K-cycle cadence (1 ms @ 10 MHz)
  vluint64_t t1 = wait_wave(12, 30000);
  CHECK(t1 < 20000, "step 1 reached within ~1 ms cadence");
  vluint64_t t2 = wait_wave(24, 30000);
  CHECK(t2 < 20000, "step 2 cadence");
  vluint64_t t10 = wait_wave(120, 130000);
  CHECK(t10 < 120000, "step 10 (top) reached");

  // down ramp: 108, ..., 0
  vluint64_t td = wait_wave(0, 130000);
  CHECK(td < 120000, "ramp down reached 0 (full cycle)");

  // DONE pulse at cycle end (loop) + RUN remains on
  CHECK((wave_status() & 2) == 2, "CYCLE set on iteration 2 (loop profile)");
  CHECK((wave_status() & 1) == 1, "RUN still on (loop)");

  // ---- ES-2.1.2: mid-cycle HOLD_MS change deferred to next cycle ----
  // currently step cadence is 1 ms; at wave=12 again (iteration 2 up),
  // rewrite WAVE_CFG with HOLD_MS=2; the NEXT step (24) must still come at
  // ~1 ms cadence; only after cycle end does the cadence become 2 ms.
  t1 = wait_wave(12, 30000);        // iteration-2 step 1
  axi_write(0x14, (1 << 0) | (2 << 2) | (12 << 10));   // HOLD_MS 1 -> 2
  vluint64_t start = cyc;
  vluint64_t t3 = wait_wave(24, 30000);
  CHECK(t3 < 20000, "mid-cycle HOLD_MS change deferred (step still ~1 ms, ES-2.1.2)");

  // ---- profile C: one-way, 14 steps, DONE sticky, RUN drops ----
  vluint64_t c_start = wave_val();
  axi_write(0x14, (3 << 0) | (1 << 2) | (4 << 10));    // PROFILE=C, HOLD_MS=1, STEP=4
  // engine re-arms from the current output; up 14 steps of 4 LSB
  vluint64_t c_top = c_start + 14 * 4;
  fprintf(stderr, "C: start=%llu top=%llu\n", (unsigned long long)c_start, (unsigned long long)c_top);
  // 14 steps at 1 ms cadence = ~140K cycles from re-arm
  vluint64_t tC = wait_wave((int)c_top, 150000);
  fprintf(stderr, "C: waited %llu cycles, last wave=%llu\n", (unsigned long long)tC,
          (unsigned long long)wave_val());
  CHECK(tC < 146000, "profile C reached top (14 steps)");
  CHECK((wave_status() & 1) == 0, "RUN dropped (one-way profile)");
  CHECK((wave_status() & 4) == 4, "DONE sticky set (profile C)");
  // a WAVE_CFG write clears DONE
  axi_write(0x14, (3 << 0) | (1 << 2) | (4 << 10));
  CHECK((wave_status() & 4) == 0, "WAVE_CFG write clears DONE");

  if (failures) {
    printf("t_waveform: %d FAILURES\n", failures);
    return 1;
  }
  printf("t_waveform: PASS\n");
  return 0;
}