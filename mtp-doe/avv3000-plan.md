# AVV-3000 "Hold the Line" — fixture + n2-vs-n3 head-to-head plan

Status: PLAN (staged implementation; written 2026-09-09 at ~100K ctx, session
thufir/ses_f783e3672 — boot by reading this file + wiki
`concepts/mtp-speculative-decoding.md` "Manual arm sweep" section).

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
      prompt-a.md prompt-b.md   # the one prompts (variant bug sets)
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

Variants a/b: rotate which of {B1,B2,B4,B5} is seeded (keep B3 both times) and
swap waveform profile (appnote profiles a vs b).

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
whole ladder must run in < 90 s. Stage-6 acceptance includes "bug-free golden
tree scores 100" (kept in seed/golden/ as the fixture's own regression).

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
  directory. Work the ticket in TICKET.md against payload/."
- Metrics per run: ladder score, wall (DB), tok/min, token-weighted tg,
  request-acceptance distribution (episode_stats), peak ctx.
- Decision rule: if |score(a) − score(b)| > 1 rung within an arm →
  nondeterminism/contamination → investigate, don't average. Else decision =
  mean wall/tok-min across variants; if within ±3% the published arm-sweep
  tie stands and we keep n2 on floor grounds; a > 5% wall win flips live config.

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

## 10. Open questions for S0 (answer before coding)

1. Keep plant sim at S5 vip tier, or drop RTL-adjacent sim to v1.1?
2. microblaze rung: -fsyntax-only fallback ok, or is the MB layer table stakes
   for realism? (varies which way it cuts scoring honesty)
3. 4 official runs (S8) enough for a call under the §7 rule, or 6 (add n3c/n1c
   to also re-verify the n1-n2 gap at 100–200K ctx, ~+3 h machine)?
```
