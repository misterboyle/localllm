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