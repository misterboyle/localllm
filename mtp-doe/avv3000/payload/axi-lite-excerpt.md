# AM004 — AXI-Lite Interface Specification, Chapter 4 (extract)

**Extract for the AVV-3000 platform.** This excerpt covers the AXI4-Lite
handshake and response rules that the ADS-8851 slave implements and that the
AXI-Lite bridge (`rtl/axi_lite_bridge.sv`) must satisfy. The ADS-8851 data
sheet §4 points here.

---

## 4.1 AXI4-Lite channel overview

AXI4-Lite is a 5-channel memory-mapped interface: write-address (AW), write
data (W), write response (B), read-address (AR), read data (R). Each channel
is a VALID/READY handshake pair:

- The source asserts VALID when it offers a transfer; the destination asserts
  READY when it accepts. A transfer occurs on the clock edge where both are
  asserted.
- The source must not wait for READY before asserting VALID (it may assert and
  hold). The destination may assert READY before VALID (legacy), but must not
  rely on it.

## 4.2 Write path

AW and W channels are independent. The slave accepts: AW first, W first, or
simultaneously; the write is COMMITTED when both the AW transfer and the W
transfer for the same write have completed. After commit the slave issues the
B (write response). Receive (burst) handling: AXI4-Lite transfers are always
single-beat; AWLEN = 0 implicit.

## 4.3 Response ordering — back-to-back writes (the invariant)

**Every committed write produces exactly one B response, in address order,
with BVALID held until BREADY. A new AW/W arriving while a previous write's B
is pending must NOT suppress, merge, or drop the pending B.** Violating this
invariant (drops of BRESP on back-to-back writes) corrupts the host's write
completion tracking: the host sees a missing response, retries, and the
register write lands twice (e.g., a WAVE_CFG.PROFILE toggle double-stepping an
AVA-QC ramp).

Compliance test: issue two writes back-to-back (second write commits while the
first B response is still pending) and require two B responses with correct
ordering. This is the `t_bus` RTL check.

## 4.4 Read path

ARVALID/ARREADY gates address acceptance; the slave returns R with RVALID
asserted until RREADY. Reads are non-destructive except where the register
map says otherwise (ADS-8851 STAT/ERR_STAT semantics, ES-2.1.3).

## 4.5 Responses and errors

- B/RESP = `OKAY` (2'b00) for legal accesses.
- `SLVERR` (2'b10) for: decode errors (reserved address, bad WSTRB), and
  write-only/read-only violations.
- `EXOKAY`, `DECERR` are not used on this interface.

## 4.6 Slave timing bounds (ADS-8851)

- Write commit latency ≤ 4 ACLK after both handshakes complete.
- B latency ≤ 6 ACLK from commit (each write, back-to-back included).
- Read data latency ≤ 4 ACLK from AR award.

## 4.7 Register-window arbitration note

AXI accesses arriving during the SDR engine's window-commit instant are
arbitrated in favor of the bus; commit stalls ≤ 2 ACLK (ADS-8851 §7.4.2 H-3).
No state is lost; the window result is held and presented once the bus is
free.