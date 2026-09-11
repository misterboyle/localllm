# AVV-3000 "Hold the Line" — fixture + n2-vs-n3 head-to-head plan

Status: PLAN AMENDED + IMPLEMENTATION IN FLIGHT (jessica, 2026-09-09 — review
amendments in §4/§5/§6/§7/§10; S0/S1/S2 built and mostly verified; see
"Progress log" at the end of this file for exact state + next steps).

## 1. Purpose

One deterministic, verifiable, ~200K-context one-shot task to settle n_max=2 vs
n_max=3 (live config = n2 per arm sweep, 2defa54) at the context lengths where
daily work actually lives (100–200K peak — past the arm sweep's 25–40K).
Secondary purpose: the reusable stage-2 DOE fixture the 2026-09-08 DOE wanted
(clean-room, no journaling leaks, single scalar score).

Success criteria:
- `score.sh` yields one scalar out of 100 from hermetic tests; no network.
- A fresh session given ONE prompt lands 150–200K peak ctx, scores meaningfully.
- Two task-variants (a/b) so each arm gets 2 non-identical runs.
- Decision rule agreed before runs (see §7): arms tie on score (MTP is
  lossless); **wall time / tok-min is the decision metric**.

Anti-benchmark posture: this is synthetic and bespoke — not downloadable, not
train-intext, bugs seeded fresh. Qwen-benchmarkedness is not a concern.

## 2. Scenario (one paragraph)

AVV-3000 arthroscopic pump = AVV-2200's big sibling. One subsystem: the ADS-8851
octal ADC + waveform-engine sensor ASIC, hung off a Zynq-style SoC: AXI-Lite
peripheral (RTL), STM32F4 safelight controller (bare-metal C), MicroBlaze BSP
(optional rung), Linux userspace telemetry service, Qt QC panel (offscreen),
all driving/observing a simulated fluidic plant (pump → line compliance →
occlusion pressure). Field ticket: *occlusion alarms fire 1-in-5000 at 37 °C;
lab can't reproduce.* Fix the seeded bugs, then implement the AVA-QC staircase
waveform pattern (inherited AVV-2200 feature) across the whole vertical slice.

## 3. Layout

    mtp-doe/avv3000/
      README.md            # runbook incl. agent rules (see §7)
      prompt-a.md prompt-b.md   # the one prompts (variant bug sets); the
                                # active prompt is copied into each run's
                                # workspace as TICKET.md (see §7 preamble)
      score.sh             # ladder runner -> scalar + per-rung JSON
      env.sh               # brew pins: verilator cmake qt qemu arm-none-eabi-gcc
      payload/
        ADS-8851-datasheet.md        # ~55K tok: reg map, SDR timing, §7 init seq
        ES-2026-04-errata.md         # 3 errata; ES-2.1.3 is the trap
        appnote-ava-qc-staircase.md  # ~12K tok: pattern spec, 3 profiles a/b/c
        axi-lite-excerpt.md          # ~18K tok: AM004 ch.4 prose+ASCII tables
        field/                       # ~28K tok: uart.log, i2c.pcap.txt, watchdog
                                     # histograms, thermal derating table, ticket
      rtl/                 # ads8851_ctrl.sv, axi-lite bridge, waveform.sv (stub)
        tb/                # verilator C++ testbenches; Makefile
      fw/
        stm32f4/           # drv_ads8851.c (the meat), startup shim, host test shim
        qemu/              # cortex-m3 semihosting runner (qemu-system-arm)
        mb/                # microblaze BSP layer — OPTIONAL rung (see §9 risk R4)
        tests/             # Unity host tests: t_fw_init, t_drdy, t_reg_arb...
      svc/                 # linux(ish) C service: ring buffer, CSV log; tests
      gui/                 # Qt6 offscreen panel + QtTest (t_gui_unit)
      sim/                 # plant.c + closed-loop tests (OPTIONAL rung)
      seed/                # SECRET: BUGS.md solution key, variant maps. gitignored!
      runs/                # gitignored; one dir per head-to-head session
      runs-archive/        # committed: score json + server log per official run

## 4. Seeded defects (solution key lives in seed/, never in agent view)

| id | layer | flavor | caught by |
|---|---|---|---|
| B1 | fw | init CFG uses rev-A bit assignment (datasheet §7.2 changed it) | t_fw_init (fails at start) |
| B2 | svc | ring overflow under 2 Hz burst — visible in field uart log histogram | t_svc_burst (fails at start) |
| B3 | fw+rtl | **errata trap**: post-fix code polls STAT twice per SDR window; ES-2.1.3 says the second read clears DRDY. Manual-following fix fails t_drdy until ES found; the double-read "hack" passes t_drdy but fails t_reg_arb timing assertion | t_drdy, t_reg_arb |
| B4 | rtl | AXI write FSM drops BRESP on back-to-back writes | t_bus (verilator, fails at start) |
| B5 | gui | counts-vs-ms unit scale — panel shows rate ×1000 | t_gui_unit |
| F1 | all | staircase waveform engine: RTL regs + fw API + sim profile | waveform/fv_* tests |

Variants a/b are POINT-BALANCED: each variant seeds the same failing rungs
{r_rtl, r_fw_host, r_svc, r_gui} — 55 pts unavailable in both (r_rtl 15 + r_fw_host
20 + r_svc 15 + r_gui 5), max achievable 45 in both, so score is like-for-like
across variants (the §7 intra-arm gate depends on this). Non-identity comes from
per-variant bug INSTANCES — B1a/B1b, B2a/B2b, B4a/B4b, B5a/B5b (B3 shared both) —
plus the waveform profile swap (appnote a vs b). seed/BUGS.md carries the 8
instances + shared B3 + per-variant map (see S6).

## 5. Scoring ladder (score.sh, 100 pts)

| rung | pts | starts failing | note |
|---|---|---|---|
| r_rtl (t_bus, t_waveform) | 15 | B4 | verilator sim |
| r_fw_host (t_fw_init, t_drdy, t_reg_arb) | 20 | B1, B3 | Unity, host build |
| r_fw_qemu (arm cortex-m3) | 10 | — | same suite under qemu |
| r_fw_mb (microblaze) | 10 optional | — | see R4 |
| r_svc (t_svc_burst, log golden) | 15 | B2 | hermetic, fixed seed |
| r_gui (QtTest offscreen) | 5 | B5 | |
| r_sim closed-loop (clamp test + fv_staircase trace) | 25 | F1 | sim/; feature payoff rung |

Rules: score.sh exit = sum of pass points; every test fixed-seed hermetic;
whole ladder TEST-EXECUTION must run in < 90 s (builds run once via `make`
outside the timer). Stage-6 acceptance includes "bug-free golden tree scores
100" (kept in seed/golden/ as the fixture's own regression).

## 6. Stages

- [x] **S0 scaffold (≈0.5 d)** — dirs, env.sh pins, empty score.sh + one
      passing smoke rung, README runbook skeleton, .gitignore (seed/, runs/).
      Accept: `./score.sh` runs, prints ladder with smoke rung, git clean.
- [ ] **S1 payload (≈1 d, THE long pole)** — write ADS-8851 datasheet (target
      50–60K tok: register map tables, §7 init sequence, SDR waveform timing
      diagram in ASCII), errata sheet (ES-2.1.3 written to contradict §7.4.2),
      appnote (profiles a/b/c), AXI-Lite excerpt, field artifacts (hand-crafted
      uart/i2c captures whose histogram implies B2, thermal table). Cross-ref
      discipline: every code-side number in S2–S5 must trace to a §/table/ES id.
      Accept: word-count gate + cross-ref grep passes.
- [x] **S2 rtl (≈0.5 d)** — peripheral + bridge + waveform stub; clean tb
      (t_bus, t_waveform); waveform engine implemented (golden). Accept:
      verilator passes on golden; after seeding, t_bus fails.
- [x] **S3 fw (≈0.5 d)** — drv_ads8851.c full golden impl; host tests; qemu arm
      runner; B3 fix path must require ES (double-read hack fails t_reg_arb —
      write t_reg_arb here). mb/ source layout only.
      Accept: host+qemu green on golden. (DONE 2026-09-10 feyd — see progress
      log; mb rung = host-compiler syntax gate, R4's clang-target-microblaze
      was impossible.)
- [x] **S4 svc+gui (≈0.5 d)** — service with ring + CSV log (golden test +
       B2-capturing burst test); Qt panel + t_gui_unit offscreen.
       Accept: green on golden, r_svc/r_gui wired in score.sh.
       (DONE 2026-09-10 piter — Qt 6.11.2 now installed, r_gui live;
       `./score.sh` first run = 75/100 with r_sim pending only.)
- [x] **S5 sim (≈0.5 d, optional-vip)** — plant.c single-file deterministic
      plant; occlusion clamp test; fv_staircase trace test vs golden waveform
      (tolerance + seed fixed). Accept on golden.
      (DONE 2026-09-10 yueh — golden `./score.sh` = 100/100, ladder exec 2.4 s;
      see progress log below. Two payload/RTL flags for S1/S6 were found
      while building — do not lose them.)
- [ ] **S6 seed (0.5 d — MICHAEL HANDS-ON, the craft)** — apply bugs per
      §4/§2-variant on top of golden per seed/BUGS.md protocol; verify each
      rung's fail-catch; variant b = rotate bugs + profile b. Accept: variant-a
      tree scores the predicted pre-run scalar; seed/golden stays 100.
- [ ] **S7 protocol (0.25 d)** — finalize prompt-a/b.md (single prompt,
      points at payload/, explicit sandbox rules §7); score ingestion into
      mtp-harvest (source tag `avv3000-<arm><variant>`); order + logging
      checklist copied into README §head-to-head. Accept: dry-run README.
- [ ] **S8 dry run + head-to-head (0.5 d setup + runs)** — one throwaway dry
      run on the assistant model (calibrate difficulty: expect golden−30..−55;
      retune payload/test strictness BEFORE official runs, then freeze).
      Official: 4 sessions, order n2a → n3a → n2b → n3b, fresh server per arm
      switch, log archive `flash-next-mtp-{n2,n3}-{a,b}.log`, episode_stats on
      each. Machine budget: 60–180 min/session (100K+ output at ~40 t/s ≈
      ~1 h, plus cold prefill).

## 7. Head-to-head protocol (decide now, run later)

- Arms: `--spec-draft-n-max 2` (live, 2defa54) vs `3` (5b382fb is the
  provenance of the n3 config; flip servers.jsonc between arms, restart at
  session boundaries only).
- Order n2a, n3a, n2b, n3b; fresh opencode session + fresh git snapshot of
  seeded tree per run into runs/<id>; assistant model = self (note self-builder
  bias in report, same as every arm-sweep caveat).
- Prompt preamble (kills the 2026-09-08 journaling leak): "Do not read or write
  ~/wiki. Do not commit. Do not touch anything outside the current working
  directory. No network access. Work the ticket in TICKET.md against payload/."
  TICKET.md IS the run's prompt-<variant>.md, copied into the workspace at
  snapshot time (the prompt file is the ticket; §3 layout).
- Run hygiene (extends the 2026-09-08 DOE rules): official runs REQUIRE the
  single-slot layout (flash-next-mtp -np 1 -c 262144 — the live config; opencode
  context 262144 with auto-compaction, so a 150–200K peak is reachable before
  compaction) and ZERO other opencode sessions / server traffic during runs.
  Dry-run (S8) acceptance includes peak-ctx landing in 150–200K; if the dry run
  lands short, retune difficulty BEFORE freezing (R3).
- Metrics per run: ladder score, wall (DB), tok/min, token-weighted tg,
  request-acceptance distribution (episode_stats), peak ctx.
- Decision rule: within an arm, |score(a) − score(b)| ≥ 10 pts →
  nondeterminism/contamination → investigate, don't average (variants are
  point-balanced by construction, §4). Else decision = mean wall/tok-min across
  variants: diff ≤ 3% → the published arm-sweep tie stands, keep n2 on floor
  grounds; diff ≥ 5% → flip live config; 3–5% → run extension pair (n2c/n3c);
  still 3–5% → keep n2 (floor grounds).

## 8. What we already reuse

- mtp-harvest/episode_stats (added 2defa54), run-mtp-doe.sh session launcher
  skeleton (workspace snapshot, smoke gate, retries), AVV-2200 prompt DNA for
  F1 (register names + profile_a verbatim — keep it byte-identical so F1 stays
  comparable to prior data), server log archive ritual.

## 9. Risks

- R1 payload authoring time — datasheet is ~half total effort; do S1 in one
  focused sitting, resist polishing §8+ of the datasheet nobody reads.
- R2 scoring brittleness — everything fixed-seed and hermetic; `score.sh`
  includes net-off guard; run laddertest in CI-ish `make -C avv3000 check`.
- R3 too easy / too hard — only S8 dry run tells; retune before freezing,
  not after.
- R4 microblaze toolchain on macOS — no brew gcc for mb. Fallback (accept now):
  mb rung = syntax/header/semantics check via clang -target microblaze
  -fsyntax-only + the ring/driver logic already covered under host+arm; keep
  10 pts only if that's stable, else drop rung and reweight r_sim 35.
- R5 my-builder bias — fixture tuning by the model being evaluated; mitigated
  by Michael owning S6 seeding + golden-tree regression under seed/ (a human
  can diff the oracle).
- R6 context blowup > 262K — runbook tells agent to consult payload/ with
  targeted greps; monitor `new prompt` lines; if a run caps, it caps (same for
  both arms — fair).

## 10. Open questions for S0 — ANSWERED 2026-09-09 (jessica, review)

1. **Plant sim: build now, at S5.** r_sim is the 25-pt feature-payoff rung (F1
   staircase in closed loop); deferring caps the ladder at 75 and kills the
   score spread the §7 rule needs.
2. **MicroBlaze: -fsyntax-only fallback accepted** (per R4). mb/ ships as a
   syntax/header/semantics check via clang -target microblaze; 10 pts
   provisional — reweight r_sim → 35 if it proves unstable.
3. **Official runs: 4** (per §7). Extension pair is n2c/n3c, NOT n1c — this
   fixture's job is the stated n2-vs-n3 call at 150–200K; n1 lives in the arm
   sweep's jurisdiction.
```

## Progress log — 2026-09-09 (jessica, implementation session)

### Review verdict (before building)
Plan is sound for its goal (settle n2 vs n3 at 150–200K ctx). Amended:
- **§4 variants are now point-balanced** — both variants seed the same failing
  rungs {r_rtl, r_fw_host, r_svc, r_gui} via per-variant bug instances
  (B1a/B1b, B2a/B2b, B4a/B4b, B5a/B5b + shared B3). Any unbalanced 2+2/3+1
  split would have tripped the §7 intra-arm gate on every run. Max achievable
  45 in both variants.
- **§7 decision band fixed** — ≤3% keep n2 / ≥5% flip / 3–5% → extension pair
  (n2c/n3c); gate threshold now |Δscore| ≥ 10 pts (not "1 rung").
- **TICKET.md resolved** — the prompt file IS the ticket (copied into the
  workspace at snapshot time); preamble adds explicit "No network access".
- **Run hygiene in §7** — official runs require single-slot layout
  (flash-next-mtp -np 1 -c 262144, live config) + zero other sessions. Live
  config verified: 1×262K slot, opencode ctx limit 262144 + auto-compaction
  → 150–200K peak reachable.
- **§10 answered**: plant sim built now (r_sim = 25-pt feature payoff); MB =
  -fsyntax-only fallback accepted; 4 official runs (extension n2c/n3c).
- Michael confirmed (2026-09-09): arms expected to tie on score (MTP lossless);
  decision metric is mechanical (tg/tpp/wall) — matches §7.

### Built this session (all in mtp-doe/avv3000/)
- **S0 scaffold**: dirs, .gitignore (seed/, runs/, build/), env.sh
  (check/install-core/install-gui tiers), root Makefile placeholder.
- **S1 payload** (contract-complete; datasheet still below the 50–60K token
  gate — needs an expansion pass):
  - `payload/ADS-8851-datasheet.md` — Rev B, §1–§11, register map §6 definitive
    (CTRL/CFG/STAT/ALARM_*/WAVE_*/ERR_STAT/IRQ_MASK/TH_RATE/SAFETY_GATE +
    data port 0x40..0x5C), §7.2 init sequence (the B1 contract: CFG=0x49,
    rev-A bit-7 trap), §7.4.2 two-read sequence (superseded by ES).
  - `payload/ES-2026-04-errata.md` — ES-2.1.3 THE TRAP (STAT read clears DRDY;
    second read/window = violation + destructive), ES-2.1.1/2 decoys.
  - `payload/appnote-ava-qc-staircase.md` — profiles A/B/C tables (§3), closed-
    loop expectation for fv_staircase, QC unit-scale rule, AVV-2200 parity.
  - `payload/axi-lite-excerpt.md` — AM004 ch.4, §4.3 the BRESP invariant (B4).
  - `payload/field/` — TICKET.md, uart.log + gen_uart.py (422 lines, gap
    summary implies B2), i2c.pcap.txt (double STAT reads → B3), 
    watchdog-histogram.txt, thermal-derating.csv.
- **S2 rtl** (golden, verilator-verified):
  - `rtl/ads8851_pkg.sv` (register constants), `rtl/axi_lite_bridge.sv`
    (golden 2-deep B FIFO — B4 lives here), `rtl/waveform.sv` (AVA-QC engine,
    profiles A/B/C + ES-2.1.2), `rtl/ads8851_ctrl.sv` (top: regs, SDR
    sequencer, ES-2.1.3 STAT semantics, occlusion/alarm, IRQ).
  - `rtl/tb/t_bus.cc` — **PASSES 16/16** on golden (count corrected
    2026-09-10; BRESP back-to-back,
    SLVERR/WSTRB, ES-2.1.3 destructiveness, ERR_STAT RC, alarm latch/clear,
    ERR_CNT).
  - `rtl/tb/t_waveform.cc` — **2 failures at last clean run** (see IN FLIGHT).
  - TB driver lesson (recorded for the fixture's own make/README): AXI master
    must capture handshake state BEFORE tick(); post-tick checks miss the
    1-cycle rvalid window (cost a debug round).

### IN FLIGHT (start here next session)
1. ~~t_waveform cfg_wr bug~~ **RESOLVED 2026-09-10 — see below. Rung r_rtl
   fully green (t_bus 16/16 + t_waveform 13/13).**
2. Then: **S3 fw** (drv_ads8851.c + host tests t_fw_init/t_drdy/t_reg_arb +
   qemu runner + mb syntax check), **S4 svc+gui**, **S5 sim** (plant.c +
   clamp + fv_staircase), **S1 datasheet expansion** to the 50–60K token gate,
   **S6 seed materials** (BUGS.md + patches + apply_variant.sh — Michael
   executes), **S7 prompts + README runbook + score.sh wiring**, then golden
   score verification (expect 95/100 sans gui; r_gui 5 deferred until qt).
3. Toolchain: verilator 5.052, qemu-system-arm, arm-none-eabi-gcc 16.2 all
   installed via brew this session (env.sh mirrors the tiers). Qt6 NOT yet
   installed (Michael: core now, qt deferred).

## Progress log — 2026-09-10 (irulan, t_waveform green session)

**Goal: the plan's IN FLIGHT unit "cfg_wr fix + t_waveform green". DONE —
r_rtl green: t_bus 16/16, t_waveform 13/13.**

### The cfg_wr fix (as diagnosed 09-09)
- `waveform.sv`: new port `cfg_data_i[31:0]`, wired to `wr_data` at the top.
  The cfg_wr branch now decodes NEW config from the write data
  (`cfg_prof/cfg_hold/cfg_step/cfg_inv`) — the register mirror NBA-updates on
  the same edge, so `*_i` ports are stale under `cfg_wr_i` (that was the bug).
- `profile_prev` (broken last-cycle hack) → `profile_applied` reg: set on
  idle start, on (re)start, on mid-run profile switch, cleared on stop.
- Mid-run switch: STEP/INVERT apply immediately, step_cnt re-arms when the
  profile changes; HOLD_MS still deferred — golden now latches
  `hold_lat <= hold_ms_i` at loop cycle-end (ES-2.1.2 literal reading).
- Restart path gated on `en_i` per datasheet §6.15 (`RUN = WF_EN & PROFILE!=0`);
  config written while disabled lands in the register and the idle-start path
  picks it up when en rises. tick_cnt resets on (re)start.

### Surprise: the 2nd failure was the TESTBENCH, not the RTL
- Trace forensics (VCD — see tooling note below) showed `cycle_o`, `iter_reg`
  and the WAVE_STATUS shadow all correct at cycle end (status = 0b111 then
  0b011). The check itself was `(wave_status() & 2) == 1` — AND-mask can only
  yield 0 or 2: **unsatisfiable, no RTL can pass it.** Fixed to `== 2`. The
  09-09 `cycle_o -> comb` change had actually worked.
- Same family, second fixture bug found by inspection: profile-C block waited
  `wait_wave(c_top, 130000)` + `CHECK(tC < 120000)` but golden needs ~140K
  cycles (14 steps x 1ms) — unreachable margins even with correct RTL. Bumped
  to 150000/146000. **Fixture README (S7) must include: mask with `== bit` not
  `== 1`, and sanity-check every timing margin against golden cadence math** —
  t_waveform had one unsatisfiable check and one unreachable margin; agents
  would have chased ghosts forever.
- 15/15 t_bus count was stale: it runs 16 checks (ERR_CNT check was added
  after the note). PASS = zero failures regardless.

### Tooling note (supersedes the 09-09 "don't burn time on VCD" note)
- `verilator --trace` on the `verilator` command line DOES add
  `verilated_vcd_c.o` to the link (5.052, macOS — the 09-09 failure was the
  flag placement, not verilator). `--trace` now in the Makefile t_waveform
  target. Debug without traces is guesswork; this bug pair (one RTL, one TB)
  was only separable from the trace.
- Trace exposes internal regs (iter_reg, st, step_cnt) at level 99 even when
  the scope header looks port-only — grep the header before giving up.
- Caveat: trace.vcd at full run length is 33 MB / ~1M cycles; fine for one
  debug round (already covered by the fixture `.gitignore` `*.vcd`).

### Next (unchanged)
**S3 fw** (drv_ads8851.c + t_fw_init/t_drdy/t_reg_arb + qemu + mb), then
S4 svc+gui, S5 sim, S1 datasheet expansion, S6 seeding (Michael), S7
prompts/README/score.sh, golden verification (95/100 expected pre-Qt).

## Progress log — 2026-09-10 (feyd, S3 fw session)

**Goal: plan unit "S3 fw" — golden driver + host tests + qemu runner + mb
rung. DONE — host 12/12, qemu (cortex-m3) 12/12, mb syntax OK, stm32f4
reference port syntax OK. `make -C mtp-doe/avv3000 check` = rtl + fw all
green.**

### Built (fw/)
- `stm32f4/ads8851_defs.h` (register contract, mirrors rtl/ads8851_pkg.sv),
  `hal_ads8851.h` (3-function port contract), `drv_ads8851.c` (golden):
  §7.2 init steps 1-8 exact incl. ES-2.1.1 ordering note, H-2 ONE-read
  `ads8851_poll_window` (ES-2.1.3 comments), LPM/safelight/alarms, F1 wave
  API (`wave_start/stop/set_hold` — set_hold implements the ES-2.1.2
  stop/rewrite/restart workaround). `ads8851_init` / `ads8851_poll_window`
  names match the ES document's references.
- `stm32f4/model_ads8851.c` — behavioral model (golden mirror of
  ads8851_ctrl.sv): window sequencer, ES-2.1.3 per-window STAT budget +
  ERR_CNT/W_VIOLATIONS, ERR_STAT RC, occl debounce, ALARM_CLR/SWRST,
  SLVERR→ALARM_COMMS, write/read event log + violation/slverr/max-reads
  observations. Deviations D1-D5 documented in the file header (D1/D2 =
  "within one SDR window" read scoping — RTL also counts pre-first-enable
  reads; no tested contract depends on that).
- Ports: `port_model.c` (host+qemu, deterministic tick-driven DRDY_N wait,
  pure pin peek — never consumes the flag), `port_stm32f4.c` (REFERENCE,
  syntax-gated only: I2C host-bridge transport matching field/i2c.pcap.txt,
  DRDY_N on PB13), `mb/ads8851_mb_bsp.c` (port for MicroBlaze fabric base
  0x4000_0000, DRDY via xgpio wait — never STAT).
- Tests (vendored Unity v2.6.0 in tests/unity/, MIT): **t_fw_init** 5 —
  exact 8-event §7.2 log, Rev-B CFG layout (bit7 message; B1 fail-catch),
  ES-2.1.1 order, step-8 DRDY, clean-error state. **t_drdy** 4 — sustained
  8-window capture, LPM freeze/resume (§7.3), jittered-edge recovery
  (§9.2), window-period from CFG. **t_reg_arb** 3 — driver=1 STAT
  read/window + ERR clean; **negative control** simulating the double-read
  hack fires STAT_VIOLATION + ERR_CNT=2 + W_VIOLATIONS + RC (liveness
  pinned per 09-10 lesson) and pins t_drdy-blindness (capture keeps
  working — why two tests exist); commit-instant H-3 access test.
  t_drdy asserts CAPTURED DATA ONLY by design (plan §4 — hack must pass it).
- Runners: `fw/tests` make run (host); `fw/qemu` — SAME test .c files
  cross-built (-std=gnu11, -Dmain rename shim, no libc: freestanding shims
  in qemu/shim/, semihosting console/exit via bkpt 0xAB, qemu.ld for
  -M lm3s6965evb), run_qemu.sh parses `RESULT rc= run= fail=` + watchdog.
  mb/Makefile syntax gate; fw/Makefile = host+qemu+mb+tests-syntax;
  **root Makefile created**: `make -C avv3000 check` = rtl + fw (plan R2).

### Decisions / deviations to review
- **R4 substitution**: `clang -target microblaze` does not exist (upstream
  clang has no MicroBlaze backend — any version). §10.2's fallback became
  host-compiler `-fsyntax-only` over bsp+driver. The rung is stable → keep
  10 pts (per plan: only reweight if unstable). If real MicroBlaze GCC
  ever lands, mb/port is the whole swap.
- **Payload amendments (same session, flagged for Michael before S6)**:
  §7.2 step 8 expectation fixed to DRDY==0 only (the old "BUSY==0" is
  unsatisfiable post-enable per §7.1/RTL — same family as the t_waveform
  doc-tension from 09-10); ES/AM docs' driver paths → fw/stm32f4/;
  ES-2.1.3 verification line no longer calls t_reg_arb "(RTL)".
- Toolchain notes: brew arm-none-eabi-gcc has **no libc/newlib** (no
  formula exists) — fw qemu builds are deliberately freestanding (this is
  why qemu/shim/stdint.h exists and why the model avoids <string.h>).
  Unity needs no libc with the three UNITY_EXCLUDE_* defines.
- No score.sh wiring yet (S7). B1/B3 fail-catches are asserted-by-construction
  + t_reg_arb's built-in negative control; S6 verifies per-rung seeding.

### Next
S4 svc+gui (svc ring/CSV + burst test, Qt deferred), S5 sim (plant.c +
fv_staircase), S1 datasheet expansion to 50–60K tokens, S6 seed materials
(Michael executes; BUGS.md must carry: B1a/B1b on the §7.2 CFG value, B3 =
Rev-A two-read-CONFIRM consume in poll_window; t_drdy catches the gate
fail, t_reg_arb catches plain extra reads), S7 prompts/README/score.sh.

### Addendum (feyd, same session — post-review coverage audit of §7.2 step 8)
- Q raised: did dropping BUSY==0 from step 8 downgrade required rigor?
  No — the observation is unsatisfiable (BUSY is low only at the close
  tick; DRDY sets one tick LATER, so they never co-report; a BUSY-poll
  implementation would spend ES-2.1.3 read budgets and fail the exact
  8-event log test). But the audit found a real hole it exposed: init's
  only REACHABLE failure branch (ERR_RESET warm-restart) was untested.
  Added test_init_rejects_warm_restart → suite now 13 (host+qemu green).
  ERR_DIRTY documented in drv as defensive per §7.2.
- mb gate upgraded: -Wconversion over bsp+drv+model+port (frontend lint —
  nearest free analogue of "compile against mb": clang has no microblaze
  target, Vivado/Vitis' mb-gcc has no macOS host, qemu has no mb machine).
  Known blind spot: host sizeof(long)≠mb32 (noted in mb/Makefile).

## System topology (agreed with Michael post-S3 — supersedes the
## "MB-as-2200-legacy-cruft" sketch; pattern ported from our real product:
## camera-head STM32+small-FPGA  ⟷  CCU beefy-FPGA/MB over framed UART)
- Pump head: STM32F4 safelight + ADS8851. The STM32 OWNS the ASIC register
  file at runtime via the I2C host-bridge (unchanged — the fw ladder and
  the B3 i2c.pcap evidence require it).
- Base controller (Zynq-style SoC): ARM Linux svc/ (telemetry, CSV) + fabric
  + MicroBlaze. The MB is the fabric CONFIG-PLANE master: bridge windows /
  gating for the ASIC space, sensor-domain reset/enable, svc telemetry FIFO
  plumbing; fallback safelight path on head-link loss (why fw/mb keeps a
  drv port under the syntax gate).
- HEAD-LINK (UART, framed): STM32 ⟷ MB. Bidirectional heartbeat (miss ⇒
  safelight app drives amber + head-lost flag into telemetry); STATUS
  frames (window SEQ, ERR_CNT/W_VIOLATIONS, alarm bitmap, wave state);
  VERSION (sw/fw); CONTROL cmds (AVA-QC wave-recipe push — the pump-domain
  CCM analogue, wave-loop/self-test toggle, OCCL_TH / I_MOT gain tweaks).
- Three-serial naming trap for the S7 README: HEAD-LINK (control) vs
  TELEMETRY CONSOLE (uart.log artifact) vs QC-panel link — never conflate.
- Small stage: **S3b head-link (≈2 h, Michael's call when)** — hlink.c on
  the STM32 side:   frame codec (SOF/len/type/seq/CRC8) + heartbeat SM + tx path behind a
  2-function hal-style port (tx byte(s) /
  rx byte with test shim injection), and **t_hlink runs on the existing
  r_fw_host + qemu rungs with a fake-peer test** (injected frames cover
  bad CRC, SOF slip/resync, heartbeat timeout → amber state) — no
  MicroBlaze execution needed to prove the UART side works, same
  shim-as-peer pattern as model_ads8851. MB end = thin counterpart,
  stays under the existing mb syntax gate. No ladder changes: t_hlink
  joins r_fw_host's all-pass contract, 0 net points.
  **BUILT 2026-09-10** — see "Progress log — 2026-09-10 (S3b)" below.

## Progress log — 2026-09-10 (S3b head-link session)

**Goal: plan stage "S3b head-link". DONE — t_hlink 12/12 host + 12/12 qemu
cortex-m3; mb + stm32f4 syntax gates cover BOTH port ends; `make -C avv3000
check` green (rtl+fw), 0 net ladder points as specced.**

Built (all under fw/):
- stm32f4/hlink.h — the wire contract (self-contained; THE reference for
  the S7 README): SOF 0xA5 | LEN(0..32) | TYPE | SEQ | payload | CRC8
  (poly 0x07 init 0x00 = CRC-8/SMBUS, check "123456789"→0xF4) over
  SOF..payload INCLUSIVE; payloads little-endian; types HB/STATUS/VER_REQ/
  VER_RSP/CTRL/CTRL_ACK + CTRL opcodes incl. AVA-QC WAVE_RECIPE push;
  STATUS = 10 B (win_seq u32, err_cnt, w_violations u24, alarm, wave_st —
  maps STAT/ERR_STAT/ALARM_STATUS/WAVE_STATUS §6 per topology note).
- stm32f4/hlink.c — golden: byte-driven decoder (SOF-hunt; errors NEVER
  tear down the link — containment table in the file header), 4-deep
  delivery queue (overflow counted, liveness still recorded), tx synchronous
  through the 2-function port in ≤8 B chunks, heartbeat SM: first poll sends
  an opening HB, any CRC-valid frame refreshes UP, timeout (150 ms, 3
  missed of 50 ms period, tunable via hlink_cfg_t) → link_up=0; the
  safelight APP owns the amber action (hlink never touches the driver —
  keeps MB reuse + syntax gate clean).
- Fake-peer trio: model_hlink.[ch] = the WIRE (inject raw frames incl.
  bad_crc, capture tx, no protocol logic), port_hlink_model.c = 5-line
  port. t_hlink parses head-TX with a HAND-WRITTEN decoder + crc oracle —
  hlink's encoder is never validated by hlink's own decoder (same
  independence trick as rtl vs fw models).
- Coverage: wire-format byte-exact + STATUS LE layout, crc8 reference
  vectors, arg boundary (37 B max frame; no poll = no HB), clean decode,
  bad-CRC drop + next-frame recovery, SOF slip (pure junk AND decoy-0xA5 —
  golden values 0x33/0x34 baked so the one-frame-loss bound is ASSERTED,
  not asserted-handwavy), seq gap + 255→0 wrap non-gap, flood containment
  (queue 4, rx_overflow counted, link still UP), HB train periodicity
  (5×LEN-0 frames in 21×10 ms polls), timeout→LOST→recover (junk does NOT
  refresh; HB keeps knocking while LOST).
- Ports under syntax gates BOTH ends: port_stm32f4.c + USART2 half (PA2/PA3
  AF7, 115200 placeholder; tests-syntax) and mb/hlink_mb_port.c over an
  axi_uartlite shape (mb -Wconversion gate now also compiles hlink.c).

Decisions/deviations:
- Freestanding rule hardened (again): struct compounds (`= (T){0}`,
  35-byte struct assign) compile to memset/memcpy calls that don't exist
  in the qemu link — hlink.c is explicit-field code, comment documents why.
  Also NULL: this codebase has no stddef ANYWHERE (shim stdint only) —
  null-pointer checks are literal `0`, tests included.
- hlink.c instance state is one static struct; time comes ONLY from
  hlink_poll(now_ms) args (systick on the head, test-supplied shim-side) —
  fixture determinism preserved.
- Ladder: r_fw_host = 4 binaries (t_fw_init 6, t_drdy 4, t_reg_arb 3,
  t_hlink 12 = 25 host); r_fw_qemu same suite; points unchanged (20/10).

Next: unchanged — S4 svc+gui → S5 sim → S1 datasheet expansion → S6 seed
(Michael) → S7 prompts/README (+ the three-serial table)/score.sh.

## Progress log — 2026-09-10 (piter, S4 svc+gui session)

**Goal: plan stage "S4 svc+gui". DONE — golden ladder `./score.sh` = 75/100
(everything except r_sim, which awaits S5). r_svc 9/9 host, r_gui 6/6
QtTest offscreen; svc_main compile-gated; root `make check` = rtl+fw+svc.**

Toolchain: Qt 6.11.2 installed via `env.sh install-gui` tier (the Qt
deferral is over — r_gui's 5 pts are live; CMake 4.3.3 was already here).

Built (svc/):
- `telem_svc.h` — THE svc contract: uart.log-canonical line format
  (byte-checked against the field sample line), signed gauge channels,
  ring cap 64 shared WINDOW/STATUS streams, and the **B2 admission
  policy** golden inverts: STATUS only while free > cap/2; WINDOW never
  lost to status; full-ring eviction only on writer stall (counted
  win_overwrite, must stay 0 on the field profile). svc_main.c = thin
  FIFO->ring->console skin, compile-only.
- `telem_svc.c` — golden; freestanding discipline kept (no FP printf —
  fixed-point %.1f by hand; no malloc/clocks; caller drives time).
- `tests/t_svc_burst.c` 5 — gen_uart.py footprint replay (15000 windows,
  5 bursts, 1 kHz writer): zero window gaps end-to-end + guard liveness
  (>25% status rejected — real pressure, seeds can't hide); burst
  accounting in+rejected == generated; saturated-ring microprobe (exact
  32-in/8-rejected guard geometry, window wins); eviction geometry;
  status console line byte-pin "#st,<ts>,<last_win_seq>".
- `tests/t_svc_log.c` 4 + `t_svc_log_ref.csv` (512 lines, --regen from
  golden, every hand-derived line present in test_excerpt_spot_checks —
  derived from run_excerpt() formulas, not copied).

Built (gui/): qc_rate.{h,c} pure-C unit-scale core (AN-8851-07 §5 ×1000
lives here EXACTLY once; integer, wrap-correct, clamped), qcpanel.cpp =
presentation-only skin (no arithmetic — B5 = any layer breaking that),
t_gui_unit.cpp 6 QtTest slots offscreen (exact 2500, truncation-once 333,
chain, u32 count wrap, clamp, panel text pins "rate: 2500 counts/s").
CMake+Makefile (QT_QPA_PLATFORM=offscreen); AUTOMOC on for QtTest.

score.sh EXISTS (S0 stub never landed; S4 acceptance demanded the wiring):
rung table name|pts|cmd|gate; PASS/FAIL/SKIP(qt)/PENDING(r_sim pre-S5);
prints per-rung JSON + "SCORE n"; exit code = score. S7 extends with
mtp-harvest ingestion, rung table is the contract unchanged.

Design decisions / deviations:
- **Writer model has no idle credit**: drain budget zeroes whenever the
  ring empties — a 1 kHz writer can't super-drain a burst with credits
  from the quiet 4.9 s (the first draft banked credit and the guard never
  fired — liveness assert caught its own fixture going no-op; same
  "harness can be the bug" family). Field math: 500/s windows alone
  drain fine; burst windows demand 2.5 kHz — overflow is burst-correlated
  exactly like uart.log.
- Channels signed int16 (aspiration negative) — parser/formatter
  round-trip on that; wire is signed.
- Test thresholds are floors/margins, not golden-arithmetic pins: seeds
  (B2a/B2b) must fail via the geometry probe + zero-gap asserts, not by
  us overfitting to one schedule. S6 verifies per-rung fail-catch.
- Three-serial naming (S7 README): svc console = telemetry console
  (uart.log artifact); not HEAD-LINK (hlink frames), not QC-panel link.
  svc consumes FIFO records in wire format; STATUS here is the svc's own
  burst, hlink STATUS frames are a different animal.

Next: S5 sim (plant.c + clamp + fv_staircase, 25 pts), S1 datasheet
expansion to the 50-60K gate, S6 seed materials (Michael executes — B2
homing notes now in telem_svc.h), S7 prompts/README/score-ingestion;
then golden verification = 100/100 (expect, post-S5).

### S4 review addendum (same day — Michael challenged the int16 change)
- Supersedes the "channels signed int16 (aspiration negative)" bullet
  above: datasheet **Table 5-1 defines the domain** — CH0/CH1 are
  0–3000 mmHg → 0..32767, unsigned. "Aspiration" in AVA-QC + INVERT
  withdrawal is pump *action*; the pressure channels never see negative
  per spec. Channels are now uint16 domain 0..32767, enforced at the
  parser (minus sign / >32767 → REJECT, no clamp/wrap). Temp stays
  signed — Table 5-1 thermistors are −40..+125 °C.
- Writing the range test found a real parser bug: a shared
  magnitude bound let positive "32768" wrap to -32768 instead of
  rejecting, and rejected the formatter's own canonical "-3276.8"
  (round-trip broken at the minimum). Bounds now sign-separated.
- New guarantee in t_svc_log: EXHAUSTIVE domain sweep (all 32768 channel
  values + all 65536 t_c10 values) through format→parse→re-format:
  value-preserving and canonical-stable. Lesson generalizes for S6/S7:
  any interface-type decision needs the domain written down where the
  code cites it (telem_svc.h now cites Table 5-1).

## Progress log — 2026-09-10 (yueh, S5 sim session)

**Goal: plan stage "S5 sim". DONE — golden ladder `./score.sh` = 100/100**
(all seven rungs PASS; full-ladder TEST EXECUTION 2.4 s, budget < 90 s).
`make check` = rtl+fw+svc+sim (sim joined the gate; gui stays Qt-gated).

### Built (sim/)
- `plant.h/plant.c` — THE plant: single state (pressure, Q16), explicit
  Euler at 1 ms, seeded LCG noise. Parameters derive from AN-8851-07 §4 in
  plant.h: DC gain 6.0012 (profile-A peak 720 counts vs the 960-count
  ALARM_CFG clamp = 25 % margin), tau 40 ms (95 % in 3·tau <= 2 x tightest
  hold; plateau-end lag <0.2 %), relief 32000 counts (below 32767
  full-scale, above every reachable OCCL_TH), noise +/-1 count, seed
  0x5eed51a7. No floats, no libc time — hermetic like the rest.
- `model_waveform.{h,c}` — C mirror of rtl/waveform.sv at ms cadence (the
  D3 "engine cadence lives in sim/" home). State machine mirrors the RTL
  branch-for-branch, incl. wave_q retention through stop, cfg_wr decode
  from write data, ES-2.1.2 deferred HOLD at cycle end.
- `port_sim.c` — hal port + world: 1 ms = wf_clock + pump-command register
  (§8.1: once the engine RUNs, the port keeps commanding WAVE_DATA — same
  value fw would write; pre-engagement the host owns the pump, which is
  how fv_occl commands the fault case) + CH0 injection + mdl_tick(10000).
  WAVE_DATA/STATUS reads intercepted before the model; WAVE_CFG writes
  fan out to model reg + mirror edge.
- `tests/fv_staircase.c` — profile A closed loop (engine→plant→CH0 via
  the REAL drv poll path): every §4 tolerance as an assertion
  (plateau-end |e|<=4, rise-95 % <= 2 x hold, overshoot +-7 on ramps,
  every sample <= 960, DONE pulses once/cycle, CYCLE, alarm quiet,
  ES-2.1.3-clean); profile C at exact app-note defaults (one-way, 14th
  plateau sticky, RUN drops, DONE sticky). Noise ON for A (seeded),
  off for C (6 % of the 24-count C plateau increment = 1.44 counts —
  +-1 LSB sensor noise cannot live inside it; noise robustness is A's
  job).
- `tests/fv_occl.c` — occlusion clamp tests: latch geometry probe (DEB=2
  -> latches exactly on the 4th OCCL_DET window close, §8.2 "DEB+1 elapsed
  windows" at next close), latch survives vent, release is ALARM_CLR-only,
  re-arm walks the debounce from zero; full-bore fault: peak rides to the
  32000 relief valve, alarm fired long before, peak <= ceiling.
- `fw/stm32f4/model_ads8851` gained the analog-input seam (additive):
  `mdl_set_ch0_in()` (feeds window latch AND the §8.2 compare; default =
  the RTL constant 42, so fw host+qemu suites unchanged — re-verified via
  ladder) + `mdl_ctrl_state()` peek (no bus, no ES budget, mdl_drdy_line
  discipline).

### Physics bugs found BY the fixture (both were mine, both deterministic)
- Q16 flow term shifted per step truncated sub-1-count/ms flows to ZERO —
  small set-points simply never pumped (the "plant is dead at sp=4" case).
- Euler ordering: orifice decay applied to p+flow instead of pre-step p
  biased the DC gain to 5.85 (-2.5 %) — ate the plateau tolerances. The
  §4 tolerances are tight enough to catch plant-model math errors, which
  is exactly what they are for.

### FLAGS for S1 / S6 (do not lose)
1. **app-note/field-range conflict**: AN-8851-07 §3 default holds (A 500,
   B 800 ms) exceed WAVE_CFG.HOLD_MS 1..255 (§6.13) — drv_wave_start
   rejects them. fv_staircase runs A at 250 ms (largest legal; §4
   tolerances are hold-relative, so they transfer). S1 must decide the
   arbiter: widen the field/scale in §6.13+RTL, or amend §3 defaults.
   The agent-facing payload currently contradicts itself — fix BEFORE S6.
2. **waveform.sv golden truncation**: `reg [7:0] holdtop_ms` clamps
   profile-B HOLD_TOP 1200 -> 176 ms; t_waveform never checks B dwell, so
   it went unseen. sim mirror follows the ARBITER (1200, app note §3) and
   says so in a comment. S6: either fix the golden RTL (+ re-run t_waveform)
   or adopt it as a seedable golden-bug candidate with a B-dwell t_waveform
   check.
3. §8.2 wording for S1: the golden geometry (RTL wire + model) is
   occ_cnt > DEB checked PRE-increment — ALARM_OCCL observably latches on
   the close that FOLLOWS DEB+1 consecutive DET windows. fv_occl pins "4th
   DET window with DEB=2"; keep the §8.2 text consistent with that.

### Where F1 homes in the seed (S6 reference)
fv_* exercises the F1 layer for this rung through the REAL driver: stub/
break `ads8851_wave_*` (§8.1/§6.13) or the §8.1 WF_EN arming path -> trace
goes flat -> accuracy/DONE/CYCLE assertions fail. RTL waveform.sv stubs
are caught by r_rtl (t_waveform); the sim mirror is device-side truth
(never a DUT), same rule as model_ads8851.c vs the fw suite.

### Next
S1 datasheet expansion to the 50-60K gate — MUST carry flags 1-3 (the
three docs must stop contradicting each other before seeding). Then S6
(Michael: B-homing notes all in-code now, incl. this entry for the sim
rung), S7 protocol, S8 dry run. Golden tree: `./score.sh` = 100 is the
regression to re-verify after any payload/RTL touch.

## Progress log — 2026-09-10 (Liet, pre-S6 flags session)

**Goal: land the S5 "FLAGS for S1/S6 (do not lose)" list — all three
payload/RTL flags resolved plus the S3 §7.2 step-8 review. Golden
`./score.sh` = 100/100 (7 rungs, ladder exec 3.0 s; budget < 90 s).**

### FLAG 1 resolved — WAVE_CFG.HOLD_MS widened to [15:2] (1..16383 ms)
- Arbiter (Michael, this session): WIDEN the field — the app note drives,
  the register follows. App-note §3 defaults (A 500 / B 800 ms) are now
  writable; §4 tolerance math untouched. Rolfed STEP_LSB→[21:16],
  INVERT→[22], rsvd→[31:23]; 14 bits also keeps hold×CLKS_PER_MS inside
  the 28-bit tick_cnt (16383×10000 < 2^28).
- Surfaces: §6.13 table, ES-2.1.2 field ref, ads8851_defs.h masks,
  ads8851_ctrl.sv slices, waveform.sv (ports + hold_lat 14-bit),
  drv wave_start/wave_set_hold ranges, drv_ads8851.h contract, t_waveform
  encodings, hlink WAVE_RECIPE contract (hold becomes u16 LE — the old
  1-byte hold was ALREADY inconsistent with the app-note defaults;
  t_hlink never pinned it), model_waveform follows via defs.h masks.
- fv_staircase keeps profile A at 250 ms — now an explicit short-hold
  choice (r_sim wall budget; §4 tolerances are hold-RELATIVE), no longer
  a workaround. S1 expansion: nothing to reconcile, just keep citing §6.13.

### FLAG 2 resolved — waveform.sv top-dwell truncation fixed + covered
- `reg [7:0] holdtop_ms` → `reg [10:0]`: profile B's HOLD_TOP=1200 fits
  (11-bit width bounds holdtop×CLKS_PER_MS inside tick_cnt; an 8-bit
  store truncates to 176 — documented in the reg comment for agents).
  model_waveform divergence retired (mirror == RTL again, comment says so).
- t_waveform gained the profile-B case: deterministic stop/restart pair
  (PROFILE=0 freezes wave_q; restart arms a fresh HOLD — no
  ES-2.1.2-deferral state inherited, no ramp/write race), 8-step ramp +
  HOLD_TOP dwell via coarse 20K-cycle polling. Lower dwell bound catches
  any width truncation (incl. 176), upper bound catches never-release.
  **15 checks now; liveness pinned:** reintroducing the width bug fails
  deterministically (line 186), clean-rebuild golden passes.
- S6 note: "holdtop-width truncation" is a *seedable r_rtl candidate* now
  that the catch exists — Michael's call in the variant map (BUGS.md).
- Tooling: t_waveform VCD is runtime-OFF by default (B dwell simulates
  ~13M cycles; default-cadence trace would be hundreds of MB) —
  `t_waveform --vcd` for debug rounds. Harness lesson (new
  "harness can be the bug" member): width-liveness re-runs need
  `make clean` — a same-second incremental rebuild surfaced a stale
  binary that briefly faked a golden failure.

### FLAG 3 resolved — §8.2/§6.11/§6.10 latch wording
Datasheet now states the observed geometry (RTL pre-increment wire +
model + fv_occl pin): alarm latches at the CLOSE of the first window
after OCCL_DEB+1 consecutive OCCL_DET windows (DEB=2 → latches on the
4th close), releases via ALARM_CLR only, re-arm restarts debounce at
zero. §6.10 DEB gloss and §6.11 ALARM_OCCL row reworded to match.

### Reviewed — §7.2 step-8 amendment (feyd's "review before S6")
Accepted: DRDY==0-only expectation (BUSY post-enable unsatisfiable per
§7.1 geometry); feyd's own addendum (test_init_rejects_warm_restart,
suite 13) closed the audit hole. No further payload change.

### Next (pre-S6 remainder)
S1 datasheet expansion to the 50–60K token gate — the three docs are
consistent now, so expansion is additive (no contradiction-carrying
left). Payload consistency gate (word-count + cross-ref grep under
tests/) NOT yet built — make it part of the S1 acceptance tooling.
Then S6 seed (Michael — BUGS.md candidate: holdtop-width variant),
S7 protocol (+ fixture README lessons: mask-equals-bit, margin sanity,
B-dwell coarse polling, clean-rebuild liveness), S8 dry run.
