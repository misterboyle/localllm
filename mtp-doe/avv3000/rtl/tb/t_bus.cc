// t_bus.cc — Verilator testbench: AXI-Lite contract + register semantics.
// Rung: r_rtl (15 pts) — verilator sim. Fails at start with seeded B4
// (BRESP dropped on back-to-back writes). Also covers ES-2.1.3 STAT
// read-destructiveness, decode errors, WSTRB rule, alarm latch/clear.
#include "Vads8851_ctrl.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static Vads8851_ctrl* top;
static vluint64_t cyc = 0;
static int failures = 0;

#define CHECK(cond, msg) do { \
  if (!(cond)) { printf("FAIL line %d: %s\n", __LINE__, msg); failures++; } \
  else { printf("  ok: %s\n", msg); } \
} while (0)

static void tick() {
  top->clk_i = 0; top->eval();
  top->clk_i = 1; top->eval();
  cyc++;
}

static void reset(int n = 10) {
  top->rst_n_i = 0;
  for (int i = 0; i < n; i++) tick();
  top->rst_n_i = 1;
}

// ---- AXI master model (full handshake, accepts B immediately) ----
static void axi_idle() {
  top->s_axi_awvalid = 0; top->s_axi_wvalid = 0; top->s_axi_wstrb = 0xF;
  top->s_axi_bready  = 1;                       // accept B promptly
  top->s_axi_arvalid = 0; top->s_axi_rready = 1;
}

static uint32_t axi_write(uint32_t addr, uint32_t data, uint32_t strb = 0xF, int hold_b = 0) {
  // issue AW+W; hold_b: do not accept B (to create back-to-back pressure)
  top->s_axi_awaddr  = addr;
  top->s_axi_awvalid = 1;
  top->s_axi_wdata   = data;
  top->s_axi_wstrb   = strb;
  top->s_axi_wvalid  = 1;
  int aw_done = 0, w_done = 0;
  while (!(aw_done && w_done)) {
    tick();
    if (top->s_axi_awready && top->s_axi_awvalid && !aw_done) { aw_done = 1; }
    if (top->s_axi_wready  && top->s_axi_wvalid  && !w_done)  { w_done  = 1; }
    if (aw_done && w_done) {
      top->s_axi_awvalid = 0;
      top->s_axi_wvalid  = 0;
    }
  }
  int got = -1;
  int timeout = 0;
  while (!hold_b) {
    tick();
    if (top->s_axi_bvalid) { got = top->s_axi_bresp; break; }
    if (++timeout > 40) break;
  }
  if (hold_b) {
    // keep B pending: deassert bready, wait for bvalid
    top->s_axi_bready = 0;
    timeout = 0;
    while (1) {
      tick();
      if (top->s_axi_bvalid) break;
      if (++timeout > 40) break;
    }
  }
  return got;
}

static void axi_accept_b() {
  // accept whatever B is pending
  int timeout = 0;
  while (1) {
    tick();
    if (!top->s_axi_bvalid) break;
    if (++timeout > 40) break;
  }
}

static uint32_t axi_read(uint32_t addr, uint32_t* resp = nullptr) {
  // AR handshake: capture comb state, tick to clock the transfer, then go.
  top->s_axi_araddr  = addr;
  top->s_axi_arvalid = 1;
  int timeout = 0;
  while (1) {
    int ready = top->s_axi_arready && top->s_axi_arvalid;
    tick();                       // transfer occurs at this edge if ready
    if (ready) { top->s_axi_arvalid = 0; break; }
    if (++timeout > 40) break;
  }
  // R data: rvalid pulses for one cycle after the AR edge.
  timeout = 0;
  while (1) {
    if (top->s_axi_rvalid) {
      uint32_t d = top->s_axi_rdata;
      if (resp) *resp = top->s_axi_rresp;
      tick();                     // clock the R accept so r_pend clears
      return d;
    }
    tick();
    if (++timeout > 40) break;
  }
  return 0xDEADBEEF;
}



static void advance(vluint64_t n) { for (vluint64_t i = 0; i < n; i++) tick(); }

// ------------------------------------------------------------------
int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  top = new Vads8851_ctrl;
  top->clk_i = 0;
  reset();
  axi_idle();

  printf("== t_bus: AXI-Lite contract ==\n");

  // ---- basic write / read round trip ----
  axi_idle();
  int br = axi_write(0x04, 0x49);                 // CFG = 0x49 (§7.2 step 3)
  CHECK(br == 0, "write CFG=0x49 -> BRESP OKAY");
  uint32_t r = axi_read(0x04);
  CHECK(r == 0x49, "read CFG back == 0x49");

  // ---- B4: back-to-back writes must each produce a B ----
  axi_idle();
  axi_write(0x00, 0x01, 0xF, /*hold_b=*/1);       // CTRL=1, keep B pending
  axi_write(0x0C, 0x1E0F, 0xF, /*hold_b=*/0);     // ALARM_CFG, accept Bs now
  axi_idle();
  // If B4 is seeded, the first write's B was dropped: the second write's B
  // appears alone. Golden: two B responses in order. Drain all pending Bs:
  int b_count = 0;
  for (int i = 0; i < 50; i++) {
    if (top->s_axi_bvalid) b_count++;
    tick();
  }
  CHECK(b_count >= 2, "back-to-back writes: >= 2 B responses (B4)");

  // ---- decode error: reserved address ----
  axi_idle();
  br = axi_write(0x30, 0xDEADBEEF);
  CHECK(br == 2, "reserved addr write -> SLVERR");

  // ---- WSTRB rule (AM004 §4.4) ----
  axi_idle();
  br = axi_write(0x04, 0x49, /*strb=*/0x8);
  CHECK(br == 2, "partial WSTRB write -> SLVERR");
  r = axi_read(0x04);
  CHECK(r == 0x49, "rejected partial write did not modify CFG");

  // ---- read-only violation ----
  axi_idle();
  br = axi_write(0x08, 0x0);                     // write to STAT
  CHECK(br == 2, "write to read-only STAT -> SLVERR");

  // ---- ES-2.1.3: STAT read destructiveness ----
  axi_idle();
  // window = 1 ms (SDR_WINDOW=0), ADC_EN on
  axi_write(0x00, 0x01);
  axi_write(0x04, 0x00);                         // window = 1 ms
  advance(10500);                                // past first close
  uint32_t st1 = axi_read(0x08);
  CHECK((st1 & 0x1) == 1, "STAT read #1 observes DRDY set");
  uint32_t st2 = axi_read(0x08);
  CHECK((st2 & 0x1) == 0, "STAT read #2 observes DRDY cleared (ES-2.1.3)");
  uint32_t es  = axi_read(0x20);
  CHECK((es & 0x2) == 0x2, "second read set ERR_STAT.STAT_VIOLATION");
  uint32_t es2 = axi_read(0x20);
  CHECK(es2 == 0, "ERR_STAT read-to-clear");

  // ---- alarm path (§8.2): threshold 0 -> immediate, latch, IRQ, clear ----
  axi_idle();
  axi_write(0x00, 0x01);                         // ADC_EN only
  axi_write(0x04, 0x00);                         // 1 ms windows
  axi_write(0x0C, 0x1000);                    // OCCL_TH=0, OCCL_DEB=0 -> CH0 42 >= 0, 1 window
  advance(21000);                                // >= 2 windows, DEB default 0
  uint32_t al = axi_read(0x10);
  CHECK((al & 0x1) == 0x1, "ALARM_OCCL latched (CH0 >= OCCL_TH)");
  CHECK(top->irq_o == 1, "IRQ asserted on latched alarm");
  axi_write(0x00, 0x01 | 0x20);                  // CTRL with ALARM_CLR
  al = axi_read(0x10);
  CHECK(al == 0, "ALARM_CLR clears latched alarm");
  CHECK(top->irq_o == 0, "IRQ deasserted after clear");

  // ---- STAT violate counter accumulation ----
  axi_idle();
  axi_write(0x00, 0x01);
  advance(10500);
  axi_read(0x08);  // first read (clears)
  axi_read(0x08);  // violation
  uint32_t st3 = axi_read(0x08);                 // ERR_CNT visible [7:4]
  CHECK(((st3 >> 4) & 0xF) >= 1, "ERR_CNT accumulated (visible on 3rd read)");

  if (failures) {
    printf("t_bus: %d FAILURES\n", failures);
    return 1;
  }
  printf("t_bus: PASS\n");
  return 0;
}