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

- [ ] **S0 scaffold (≈0.5 d)** — dirs, env.sh pins, empty score.sh + one
      passing smoke rung, README runbook skeleton, .gitignore (seed/, runs/).
      Accept: `./score.sh` runs, prints ladder with smoke rung, git clean.
- [ ] **S1 payload (≈1 d, THE long pole)** — write ADS-8851 datasheet (target
      50–60K tok: register map tables, §7 init sequence, SDR waveform timing
      diagram in ASCII), errata sheet (ES-2.1.3 written to contradict §7.4.2),
      appnote (profiles a/b/c), AXI-Lite excerpt, field artifacts (hand-crafted
      uart/i2c captures whose histogram implies B2, thermal table). Cross-ref
      discipline: every code-side number in S2–S5 must trace to a §/table/ES id.
      Accept: word-count gate + cross-ref grep passes.
- [ ] **S2 rtl (≈0.5 d)** — peripheral + bridge + waveform stub; clean tb
      (t_bus, t_waveform); waveform engine implemented (golden). Accept:
      verilator passes on golden; after seeding, t_bus fails.
- [ ] **S3 fw (≈0.5 d)** — drv_ads8851.c full golden impl; host tests; qemu arm
      runner; B3 fix path must require ES (double-read hack fails t_reg_arb —
      write t_reg_arb here). mb/ source layout only.
      Accept: host+qemu green on golden.
- [ ] **S4 svc+gui (≈0.5 d)** — service with ring + CSV log (golden test +
      B2-capturing burst test); Qt panel + t_gui_unit offscreen.
      Accept: green on golden, r_svc/r_gui wired in score.sh.
- [ ] **S5 sim (≈0.5 d, optional-vip)** — plant.c single-file deterministic
      plant; occlusion clamp test; fv_staircase trace test vs golden waveform
      (tolerance + seed fixed). Accept on golden.
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
  - `rtl/tb/t_bus.cc` — **PASSES 15/15** on golden (BRESP back-to-back,
    SLVERR/WSTRB, ES-2.1.3 destructiveness, ERR_STAT RC, alarm latch/clear,
    ERR_CNT).
  - `rtl/tb/t_waveform.cc` — **2 failures at last clean run** (see IN FLIGHT).
  - TB driver lesson (recorded for the fixture's own make/README): AXI master
    must capture handshake state BEFORE tick(); post-tick checks miss the
    1-cycle rvalid window (cost a debug round).

### IN FLIGHT (start here next session)
1. **t_waveform cfg_wr bug — diagnosed, fix written in head, not yet applied:**
   the engine's cfg_wr branch samples profile_i/step_lsb_i from wave_cfg_reg,
   which the register block updates with NBA at the same edge → the engine
   sees STALE config at the cfg edge. Evidence: profile geometry switches to C
   (up 14) but STEP stays 12 and step_cnt never re-arms (wave 24→168 = 14
   steps × 12). FIX: pass wr_data into the waveform module as cfg_data_i and
   use the write data (profile = data[1:0], step = data[15:10], etc.) inside
   the cfg_wr branch; track applied profile in a reg for change detection.
   Also verify the cycle_o → comb (iter_reg) change made this session (was
   the 2nd failure — expected fixed, not yet re-run).
   VCD tracing for this debug was blocked by verilator --trace not adding
   verilated_vcd_c to the link — don't burn time on it; the diagnosis above
   is solid.
2. Then: **S3 fw** (drv_ads8851.c + host tests t_fw_init/t_drdy/t_reg_arb +
   qemu runner + mb syntax check), **S4 svc+gui**, **S5 sim** (plant.c +
   clamp + fv_staircase), **S1 datasheet expansion** to the 50–60K token gate,
   **S6 seed materials** (BUGS.md + patches + apply_variant.sh — Michael
   executes), **S7 prompts + README runbook + score.sh wiring**, then golden
   score verification (expect 95/100 sans gui; r_gui 5 deferred until qt).
3. Toolchain: verilator 5.052, qemu-system-arm, arm-none-eabi-gcc 16.2 all
   installed via brew this session (env.sh mirrors the tiers). Qt6 NOT yet
   installed (Michael: core now, qt deferred).
