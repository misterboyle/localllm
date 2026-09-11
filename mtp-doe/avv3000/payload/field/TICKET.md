# Field Ticket AVV-FT-4471 — Occlusion alarms fire 1-in-5000 at 37 °C; lab cannot reproduce

**Reporter:** Apex Med device engineering, field service (tier-2)
**Date opened:** 2026-05-19 · **Status:** open
**Platform:** AVV-3000 arthroscopic pump, production image 3.2.1
**Subsystems involved:** ADS-8851 sensor ASIC + safelight controller (STM32F4),
pump telemetry service, QC panel

## Symptom

During clinical use (ambient 35–39 °C, pump head at 37 °C typical), the
safelight controller occasionally — on the order of 1 in 5000 infusion
sessions — flashes the RED occlusion alarm with no corresponding pressure
event. The QC panel shows a rate spike ×1000 times nominal for ~2 s, then the
alarm clears by operator reset. Field service cannot reproduce on the bench at
22 °C; the lab has run the unit continuously for 96 h with the occlusion
threshold crossed deliberately and observed zero false alarms.

## Evidence attached

- `uart.log` — pump telemetry service UART dump from the affected unit (3 h),
  including the 2 Hz status-burst cadence of the telemetry ring
- `i2c.pcap.txt` — I²C bus sniff of the safelight controller accessing the
  sensor ASIC through the host bridge (same unit, 20 min window)
- `watchdog-histogram.txt` — controller watchdog reset histogram vs head
  temperature bin
- `thermal-derating.csv` — ADS-8851 ADC accuracy vs ambient temperature
  (head temp 37 °C row highlighted)

## Initial analysis (field service)

- The alarm latch source is `ALARM_STATUS.ALARM_OCCL`; the occlusion compare
  uses the averaged CH0 pressure window (§8.2). No transient in the pressure
  trace accompanies the alarm.
- The rate-spike ×1000 appears to be a units-scale artifact in the QC panel
  presentation layer.
- Suspected interaction between the ASIC read path and the telemetry service's
  ring buffer under the 2 Hz burst cadence; the uart.log histogram shows
  recurring sequence gaps during bursts.
- Temperature dependence points at the ASIC's ADC/thermal behavior or a
  firmware path that only misbehaves intermittently (the lab runs cold).

## Ask

Fix the root cause(s). The vertical slice (driver, service, QC panel, and the
closed-loop plant verification) must pass the full score ladder with the
occlusion monitor behaving per the datasheet/errata contract.
## Update 2026-05-27 (field service, after evidence pass 1)

- `i2c.pcap.txt` extended with CAPTURE 2 (bench unit) and CAPTURE 3 (the
  affected unit at two evening events). The STAT access pattern differs by
  controller build, not by unit — see the per-capture analysis notes.
- `uart-image-304.log` attached: pre-deploy control capture of the same
  slot from image 3.0.4. Zero sequence gaps, zero ring overflows. The
  window-gap signature appeared in this exact slot at the 3.1.0 image
  step (the STATUS burst feature), hardware untouched throughout.
- `fleet-gap-survey.csv` attached: 12 units across two clinic sites, same
  image family. Gap sizes 12..19 everywhere they occur, burst-correlated
  everywhere, ZERO gaps on the two units still held at 3.0.x images.
- `watchdog-histogram.txt` extended: sister units at the same site show
  the same 37 °C-biased reset histogram with the same image.

## Update 2026-06-13 (field service, after evidence pass 2)

- Field-service conclusion revised on the bench: the red-safelight latch
  itself behaves per the datasheet (a genuine debounce-basis event exists
  in silicon at those instants). What fails is the instrumentation AROUND
  it: the host never gets a complete, non-destructively-read record of the
  trip windows, and the console loses neighbouring windows to its own ring
  at burst instants. See `i2c-annotation-4471.md` and
  `report-telemetry-ring-analysis.md` for the working notes, plus
  `thermal-analysis-appendix.md` for why the thermal-accuracy theory was
  closed without further work (6σ margin budget at 37 °C is not a
  false-alarm budget by two orders of magnitude).
- QC panel ×1000 rate spike now treated as a SEPARATE defect (display
  layer), see `qc-panel-rate-spike.md`. It rides along in the alarm
  window because the panel samples the same console feed; do NOT let it
  pull the root-cause hunt — it reproduces at 22 °C on demand while the
  alarm never does.
- Lab requests closed: chamber sweep at 37 °C with full console load
  reproduces (protocol in `bench-repro-protocol.md`); the cold bench
  cannot reproduce because the burst/stall coincidence rate collapses at
  22 °C (drain is never slow, ring never sits full at burst onset...
  wait — it STILL overflows every other burst; what collapses at 22 °C is
  the watchdog path. Nail this in analysis, do not hand-wave.)
- RMA-4482 (`rma-4482-no-fault-found.txt`): no fault found, unit back in
  service. Not related to this ticket beyond the scary transport story.

## Ask (unchanged)

Root-cause and fix the fault families this evidence points at, keeping the
datasheet/errata contract intact: the occlusion monitor keeps its latch
geometry, the read path keeps its ES-2.1.3 semantics, the telemetry
service keeps its line format (byte-identical to `uart.log`), and the QC
panel shows counts/sec exactly once.
