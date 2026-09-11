# AM004 — AXI-Lite Interface Specification, Chapter 4 (extract)

**Extract for the AVV-3000 platform.** This excerpt covers the AXI4-Lite
handshake and response rules that the ADS-8851 slave implements and that
the AXI-Lite bridge (`rtl/axi_lite_bridge.sv`) must satisfy. The
ADS-8851 data sheet §4 points here.

**Scope of this extract.** Chapter 4 of AM004 in full is longer; the
sections carried here are the ones cited from the AVV-3000 payload and
source tree, carried with enough surrounding text that a reader needs no
other volume to implement or verify a compliant slave, plus the
informative material (this extract's additions are marked *informative*)
that the first-time bridge implementer otherwise learns by debugging.

**Reading rules.** "MUST" / "MUST NOT" mark normative rules — a design
that violates one is non-compliant regardless of what it simulates to.
"MUST NOT" in this chapter includes partial violations: one dropped
response out of a million is the same violation as one out of one.
Tables marked ILLEGAL enumerate behaviours the specification forbids by
enumeration as well as by rule. *(Informative.)*

---

## 4.1 AXI4-Lite channel overview

AXI4-Lite is a 5-channel memory-mapped interface: write-address (AW),
write data (W), write response (B), read-address (AR), read data (R).
Each channel is a VALID/READY handshake pair:

- The source asserts VALID when it offers a transfer; the destination
  asserts READY when it accepts. A transfer occurs on the clock edge
  where both are asserted.
- The source must not wait for READY before asserting VALID (it may
  assert and hold). The destination may assert READY before VALID
  (legacy), but must not rely on it.

### 4.1.1 The handshake, normatively

For every channel (xVALID/xREADY pair):

1. A transfer takes place on a rising clock edge at which xVALID and
   xREADY are both HIGH — that edge is the transfer. Edges with
   xVALID HIGH and xREADY LOW are waits, not failed attempts.
2. Once asserted, xVALID MUST remain asserted until the transfer
   completes (the edge where xREADY is HIGH). Withdrawing a VALID
   before the handshake is ILLEGAL. *(The rationale: the destination
   may have sampled the payload on precisely that edge; a withdrawn
   offer is an uncommunicated retraction.)*
3. While xVALID is asserted, all payload signals of the channel
   (address, data, strobes, write) MUST remain stable until the
   transfer. An address that changes mid-offer is a *different,
   illegal* transaction.
4. xREADY MAY be assertive before xVALID, LOW and rise with it, or
   toggle during a wait; there is no minimum assert width other than
   "covers a transfer edge", and no dependence rule: a destination
   MUST NOT be designed such that its READY behaviour assumes any
   particular VALID timing, and a source MUST NOT depend on READY
   arriving at all without remaining able to hold VALID indefinitely.
5. Deadlock avoidance is the shared responsibility: a destination that
   refuses a channel forever (READY never asserting) may only do so in
   ways that cannot propagate back to its own source-side channels.
   Section 4.9 states the concrete ordering requirement this creates
   for endpoints that respond to writes.

### 4.1.2 Valid/ready legality matrix *(informative companion)*

|  | VALID | READY | transfer? | legal? |
|---|---|---|---|---|
| 1 | 0 | 0 | no | yes |
| 2 | 0 | 1 | no | yes (legacy assert-first; the destination may do this but MUST NOT assume a source reacts to it) |
| 3 | 1 | 0 | no | yes — and now VALID holds until case 4 |
| 4 | 1 | 1 | YES | yes — the one and only way a transfer happens |
| 5 | 1→0 while READY was 0 | — | — | **ILLEGAL** (rule 2: withdrawal after offer) |
| 6 | payload changed while VALID held | — | — | **ILLEGAL** (rule 3: stability) |

Row 5 is the source-side violation that simulation checkers catch
first on new masters; row 2 is why reset timing of VALID/READY matters
for gate-level: READY-before-VALID is legal to SEE and illegal to
NEED. Compliance questions about "did the bus see my write" always
reduce to: was there an edge of the row-4 kind, and was the payload of
row-6 kind stable through it?

### 4.1.3 Channel counts and independence

There are five and only five channels on a Lite endpoint: AW, W, B,
AR, R. There is no burst, no ID, no locking, no user lanes, no atomics
channel — any Lite transaction list, state machine, or FIFO sizing
reasoning that has to mention those is reasoning about full AXI4, not
this specification. Channels are point-to-point and unidirectional;
each has independent handshake state. Sections 4.2 and 4.6 state how
the address/data channels combine into transactions; nothing else in
this chapter crosses channel groups, with the single exception of the
write-response retention rules of §4.5.3, which exist precisely because
the write-request channels and the write-response channel are NOT
required to move in lockstep.

### 4.1.4 Widths and placement in this system

The AVV-3000 instantiation is 32-bit address, 32-bit data, one
outstanding transaction per request channel pair as a host-side policy
(datasheet §4, D.5), single clock domain (ACLK), no clock conversion
inside the bridge. Data placement is 32-bit little-endian: register
data occupies DW[31:0] on R/W; WWSTRB is 4 bits, one strobe bit per
byte lane. The register space of the attached peripheral is 4-byte
aligned — every address presented on AW/AR is a multiple of 4
(non-aligned Lite addresses have no defined target register and are a
decode error per §4.7's enumeration).

### 4.1.5 Reset interaction

During endpoint reset (RESET_N asserted, or SWRST self-clearing
window on the ADS-8851), the endpoint's acceptance behaviour is:
READYs on request channels MAY be held LOW indefinitely — the host's
obligation to hold VALID and payload unmodified continues unchanged
(§4.1 rule 2 has no reset exception) — and any half-transaction
already committed before reset keeps its response obligation only to
the extent the register map grants across reset (ADS-8851: pending-but-
unconsumed responses are lost by SWRST as documented; the §4.3
invariant's domain is a reset-stable window). The bus does NOT define
"the reset ate my write" semantics beyond that documented scope: hosts
re-initialise host-side tracking after any reset they issue, and the
verification that this was done is the datasheet init walk (§7.2),
which exists precisely because the bus gives no way to ask "what did
I acknowledge before you reset?" — the answer is always "assume
nothing; walk the config.",

*(Informative.)*

---

## 4.2 Write path

AW and W channels are independent. The slave accepts: AW first, W
first, or simultaneously; the write is COMMITTED when both the AW
transfer and the W transfer for the same write have completed. After
commit the slave issues the B (write response). Receive (burst)
handling: AXI4-Lite transfers are always single-beat; AWLEN = 0
implicit.

### 4.2.1 Pairing, ordering, and the commit point

Because AW and W progress independently, a slave MUST contain logic
that pairs them — and pairing has an ordering obligation: when a master
issues writes in AW-address order A, B and their data arrive in the
same order, the committed pairings are (A-address, A-data),
(B-address, B-data). A slave whose internal queues reorder — for
example, one that pairs a late-arriving earlier address with an already
buffered later datum — is corrupting the stream (the corruption is
silent: both writes complete with OKAY).

The pairing rule this specification chooses is FIFO order on both
channels (first address pairs with first data). A slave MAY implement
simultaneous-transfer fast paths (address and data accepted on the same
edge commit immediately) provided the fast path is observationally
identical to the FIFO. What a slave MUST NOT do is accept addresses
onto an AW FIFO while dropping, merging, or reordering entries under
any internal pressure condition — see §4.9 for the outstanding-transaction rules that bound
how deep such pressure can be, and §4.5.3 for why RESPONSES in
particular can never be merged even under pressure.

The COMMIT point (both halves received) is the point of no return:
after commit the write is a fact, its effects (a register update, per
the register's access class in the peripheral map) have happened or are
guaranteed to, and exactly one B response is owed (§4.3, §4.5). Before
commit, nothing has happened visibly, and any number of aborted AW/W
*half* pairs are legal as far as the bus is concerned.

### 4.2.2 Legal arrival interleavings *(informative)*

| order of arrival | slave behaviour | B issued |
|---|---|---|
| AW then W | hold AW, pair at W commit | once, per §4.3 timing bounds |
| W then AW | hold W, pair at AW commit | once |
| AW and W same edge | commit immediately | once |
| AW, second AW before first W (different addresses) | legal — accept and hold both in order (or back-pressure AW legitimately per §4.9) | twice, in order |
| AW then W for write 1, AW 2 held — then W 2 | pair (1,1) at W1, commit (2,2) at W2; no interleaving of pairs | twice, ordered |

### 4.2.3 What "committed" is not

Commit is not B, not "the response was received by the host", not
"posted". A host that treats AW+W acceptance as "the device has the
value" is reading the bus as if it were a wire; commit is the moment the
device guarantees persistence of the effect and the issuance of
exactly one response. Every completion-tracking bug in the Lite
family's field history (see §4.3.2) is some layer making some other
event the "real" completion signal.

### 4.2.4 Request-side buffering patterns *(informative)*

Where AW/W pairing lives determines what the endpoint can see:

1. **No buffering (combinational accept)** — AWREADY and WREADY both
   gated on "the other channel is ready too". Legal and simplest, but
   it forces AW-and-W arrival coupling (§4.2's independence promise
   survives only as slow acceptance) and it makes §4.5.3's back-
   pressure path the ONLY way to ever hold a request — fine until it
   isn't: the response queue is then the sole elasticity on the write
   path and every B stall walks straight onto AWREADY/WREADY.
2. **Address-before-data (AW FIFO, W waits)** — the historical default
   and the source of the classic half-pair deadlock when the data
   path back-pressures a "committed" address: legal only under the
   rule that an accepted-but-unpaired AW is ALWAYS retrievable toward
   commit (never silently cancelled), §4.2.1's no-drop clause.
3. **Both-channel elastic (the AVV-3000 golden bridge)** — small FIFOs
   on AW and W (depth ≥ the §4.10 envelope), pairing in FIFO order,
   commits fire independently of response state, response stalls never
   touch request-side acceptance until the §4.5.3 resource bound is
   reached and the envelope makes that unreachable-in-traffic. Cost:
   a few dozen flops. Benefit: every §4.5.1/§4.5.3 property holds by
   construction instead of by luck, and the negative tests (C8) pass
   without special-casing.

Pattern choice is an implementation decision this chapter does NOT
normalize beyond the rules; the patterns are recorded so that a
reviewer can ask the right question — "which of my three ready
signals would you like to hold right now, and does holding it lose
anything?" — and hear a correct answer.
---

## 4.3 Response ordering — back-to-back writes (the invariant)

**Every committed write produces exactly one B response, in address
order, with BVALID held until BREADY. A new AW/W arriving while a
previous write's B is pending must NOT suppress, merge, or drop the
pending B.** Violating this invariant (drops of BRESP on back-to-back
writes) corrupts the host's write completion tracking: the host sees a
missing response, retries, and the register write lands twice (e.g., a
WAVE_CFG.PROFILE toggle double-stepping an AVA-QC ramp).

Compliance test: issue two writes back-to-back (second write commits
while the first B response is still pending) and require two B
responses with correct ordering. This is the `t_bus` RTL check.

### 4.3.1 The shape of the failure *(informative)*

The canonical non-compliant bridge is a slave that answers the current
write with "the" BRESP register — a single response slot, overwritten
by whichever commit fires while the host has not yet consumed the
previous response. Its golden behaviour is perfect with any pause
between writes. Back-to-back, the trace reads like this (schematic):

    ACLK edge:   n   n+1   n+2   n+3   n+4 ... n+20
    write #1:  commit
    pending B:      [B1]  .......................   BVALID set, host
    write #2:                             commit    not yet ready
    pending B:                        ...  [B2]     <-- B1 OVERWRITTEN
    host side: one response observed; write #1 tracked as incomplete
               forever (or retried by policy): double-lands the value.

Nothing on the bus "looks wrong" at edge n+20 — the responses that
were issued were legal; the crime is the missing one. This is why
the invariant is stated as a per-write obligation ("exactly one") and
why the compliance test is back-to-back with the response-side
handshake deliberately stretched: it is the only trace shape that can
distinguish "holds one B" from "holds B's" architectures.

### 4.3.2 Retry interaction — why response bugs are data bugs

A bus whose responses can vanish cannot be layered with a retrying
host without corruption, and every real host stack retries something
(a timeout on a stuck write, a power-management resume re-push, a
config driver that "makes sure"). The invariant therefore has a
second-order safety meaning: hosts implement "retry on no-response"
because SOME buses lose responses; a device that drops BRESP converts
the host's error-recovery path into a write-amplification path. The
AVV-3000 example in the invariant paragraph is exactly this shape:
the WAVE_CFG write that doubles a profile transition is not exotic —
it is what happens when a config path with retry semantics meets a
slave whose B queue is one deep and can silently drop on overwrite.

Section 4.9 states the outstanding-transaction rules; a compliant,
flow-controlled system CANNOT lose a committed write's response, but
that is a theorem of the rules of §4.5.3 and §4.9 TOGETHER — slaves
that "can't overflow because of flow control" while still containing
destructive response slots are relying on the host for compliance,
which this standard permits nowhere.

### 4.3.1 notes for formal verification *(informative)*

The invariant decomposes into four properties the RTL sign-off
carries:

- **P1 (existence):** every commit generates a BVALID raise within the
  §4.8 bound — no committed write without a response.
- **P2 (uniqueness):** no two commits share one response transfer; no
  response transfer without a preceding commit. (Biconditional:
  responses and commits are in bijection.)
- **P3 (stability):** while BVALID is high, BRESP is constant and the
  pending queue's order does not change.
- **P4 (progress):** if the host ever presents BREADY, the pending
  head transfers within one cycle (no artificial idling past the
  bound; back-pressure lives on the REQUEST channels exclusively —
  the last clause of §4.5.3 is what makes the design's blocking
  behaviour legal to place where verification finds it).

Model-checking P1–P4 against the queue implementation takes minutes;
the interesting construction is the *negative* model whose queue has a
bypass that can overwrite the head: every trace where a one-slot
design loses a response shows up as a P2 falsifying witness, which is
why the suite drives commits against held BREADY as aggressively as it
drives ordinary traffic. The AVV-3000 regression t_bus carries C2
(which is P2's two-writes-back-to-back instance) as its namesake
check.

---

## 4.4 Write data: WDAT and WSTRB

The write-data channel carries DW[31:0] and WSTRB[3:0]. Rules:

1. Data validity is per strobe bit: bits of DW whose WSTRB bit is LOW
   are not part of the transfer and MUST be treated as absent by the
   destination (never as zero, never as prior value — as absent; see
   rule 4 for registers).
2. WSTRB with value 0x0 is a legal transfer that writes nothing. It is
   response-bearing: a zero-strobe write commits (it is a no-op with
   an OKAY response), it consumes a position in ordering, and it MUST
   NOT be silently discarded from response counting by intermediates.
3. WLAST exists on full AXI4 only; Lite writes are single-beat and the
   implicit WLAST is always asserted at the beat. Bridges that model a
   WLAST input carry it tied; nothing in this chapter branches on it.
4. **On this platform (ADS-8851; normative for AVV-3000 hosts):** the
   register file is strictly 32-bit; a write is defined over ALL FOUR
   byte lanes. A partial WSTRB (nonzero, ≠ 0xF) on a write to a
   register address is a platform-defined write error: the slave MUST
   respond SLVERR (Table 6-3, datasheet §6.3 — invalid-WSTRB class) and
   the register MUST be unmodified. The rejected write still consumed
   a transaction and a response slot: exactly one SLVERR-bearing B is
   owed, in order, per §4.3.

The platform chooses full-strobe-only over per-byte register updating
because registers are transactional (a half-written WAVE_CFG is not a
configuration state the hardware defines); the alternative would
require byte-lane semantics for every field, which the register map
does not carry. Host software therefore treats writes as whole-word:
read-modify-write is the composition tool, not partial strobes — the
one legal use of partial WSTRB in AVV-3000 traffic is NONE; the
checker in the compliance suite REJECTS partial strobes precisely so
driver code does not discover the platform-defined behaviour by
accident during development of a feature that "seemed to work".

### 4.4.1 WSTRB legality summary *(informative)*

| WSTRB on register access | bus-level class | ADS-8851 result |
|---|---|---|
| 0xF (full) | ordinary write | register updated; B OKAY |
| 0x0 | legal no-op write | register untouched; B OKAY |
| 0x1..0xE (partial) | legal AXI, platform error | register UNTOUCHED; B SLVERR (write suppressed, response in order) |

### 4.4.2 Why SLVERR-and-suppress, not SLVERR-and-partial-apply

Two ways a slave could honour "error but partially land the write"
both fail: applying lanes the address decoded for while erroring
leaves a field-split register state between the old and new value
(undefined configuration territory), and applying nothing but
reporting OKAY lies about the transaction. The standard's only honest
option for a transactional register is error + suppression, and this
platform mandates it. The compliance test pins it as a pair: partial
strobes are rejected AND the target register reads back its old value
(§4.11 checklist item C4).

---

### 4.4.3 Full-word discipline outside the driver *(informative)*

Memcpy/DMA-style engines that default to byte-lane writes on small
transfers eventually hit the §4.4 platform rule — on the AVV-3000
topology the register space is never a DMA target (the data port is
host-polled by design, D.4), but config pushes over a generic fabric
layer that "optimises small writes" has produced real tickets
elsewhere: the symptom is register writes that work individually and
fail as split sequences under a fast path. The host-side rule is flat:
every transaction that touches the endpoint's address range is one
4-byte whole-word access, regardless of how many bytes of it the
software "actually wanted". The endpoint's SLVERR is not a hint to
retry smaller — the response generation table (§4.7.1) has no row
where partial-later succeeds.

---

## 4.5 The write-response channel (B)

### 4.5.1 Handshake rules

BVALID/BREADY follow §4.1 with the payload being BRESP only. The
slave-side source rules (BVALID) are restated because this is the
channel the ADS-8851 bridge exists to protect:

1. The slave MUST assert BVALID as the unique response to each
   committed write (§4.3), within the latency bound of §4.8.
2. Once asserted, BVALID MUST hold until the edge on which the host
   presents BREADY. A response that is "held for a while and then
   cancelled because something better came along" is rule-2's
   violation and the response-loss mechanism of §4.3.1 at the same
   time.
3. BRESP MUST be stable while BVALID is held. A response code that
   changes mid-offer is ILLEGAL — including changing from OKAY to
   SLVERR when a late-detected error is discovered after commit; the
   decision tree in §4.7 fixes the code at issue time.
4. BVALID deasserts the cycle after a accepted transfer or stays to
   offer the next queued response (back-to-back offerings are legal
   provided each held its own transfer edge).

### 4.5.2 Ordering

B responses appear in commit order, which by §4.2.1 (plus the host-side
single-outstanding conventions of §4.10) is the host's write issue
order. "In address order" and "in commit order" coincide on this
platform because the host is required not to run two writes to the
same endpoint simultaneously (datasheet §4 host-side rule; the D.5
internal view), and the bridge enforces it. A multi-slave crossbar
preserves per-slave ordering without any obligation toward
cross-slave ordering — responses to different endpoints have no
relative order at all, and hosts MUST NOT infer one.

### 4.5.3 Retention of pending responses

**A compliant write-response endpoint MUST hold every generated
response until its transfer, regardless of what the request-side
channels do afterwards.** Concretely: after write #1 commits and its B
is asserted-but-unconsumed, the endpoint's obligations are:

- B1 keeps being offered until consumed (never cancelled, never
  replaced by the offering of B2, §4.5.1 rules 2–3);
- B2 is GENERATED at its own commit time or queued for it, with the
  queue sized for all responses that can be generated before the
  oldest is consumed under the §4.10 outstanding-rule envelope;
- requests arriving against an exhausted response-retention resource
  MUST be handled by back-pressuring their acceptance (holding AWREADY
  / WREADY LOW so no further commits happen), NEVER by dropping or
  merging a pending response, and NEVER by responding with a merged
  or stale code.

This clause is the datasheet-citable form of the invariant (§4.3) in
resource terms — it is the sentence an RTL implementation is checked
against, and the reason the ADS-8851 bridge's response buffer is a
queue at all: the alternative architectures (overwrite-able slot,
combined-slot for merged outstanding writes) satisfy §4.5.1 while
violating this section under exactly the back-to-back conditions
§4.3.1 constructs.

### 4.5.4 Sizing the retention resource *(informative)*

For the AVV-3000 topology (host issues ≤ N = 2 unreturned writes —
"one outstanding with the next already AW/W-queued" is the observed
driver shape), a 2-entry response queue absorbs the full request-side
pipeline while keeping the §4.5.3 back-pressure path "impossible to
see, but correct" — no legal trace stalls AW on this bus before the
queue would have to demonstrate back-pressure, and the compliance
test constructs the illegal-exceeding-N shape to prove the response
path survives it anyway. The rule for any other topology: retention
depth ≥ deepest commit-before-previous-response-consumed condition
the request rules allow; if that quantity is zero (strict serial
hosts), the depth requirement is vacuous; if it is > 0 (any pipelined
host), the queue must be at least that deep.

### 4.5.5 BRESP codes

Codes and their generating conditions are §4.7's; on this endpoint:
OKAY and SLVERR only (with §4.10's exception list: DECERR is not
generated by this endpoint; the crossbar DECERRs addresses this
endpoint never sees). EXOKAY is never generated (no exclusive-access
state exists on the endpoint, and Lite carries no AxLOCK).

### 4.5.6 Response-queue microarchitecture *(informative — the AVV-3000 golden bridge)*

The AVV-3000 golden implementation (`rtl/axi_lite_bridge.sv`) carries
the §4.5.3 retention resource as a 2-deep FIFO of BRESP entries with a
commit-side push and a transfer-side pop, head-gated BVALID —
deliberately dumb, because every cleverness at this point in the path
has historically been a §4.3 violation wearing a performance hat:

- **"Merge same-address responses"** — merges writes, which is to say
  drops one from the bijection (P2). Two writes to WAVE_CFG are two
  configuration events (arm, then re-arm) even against the same
  register in the same slot; the response count is a fact about
  transactions, not about payload entropy.
- **"One deep, oldest dropped, BREADY was never slow"** — field-
  proven wrong; see §4.3.2's retry paragraph and the seeded-defect
  history the t_bus check exists for.
- **"Separate OKAY/SLVERR channels"** — reorders, violating §4.5.2;
  response code is payload, order is the wire.

The queue's depth (2, matching the §4.5.4 derivation of the AVV-3000
envelope) is a *resource*, not a *permission*: §4.5.3's back-pressure
clause remains implemented and remains tested (checklist item C8)
even though the envelope says it should never fire. A queue that
relies on never filling is a destructive-slot design with better
optics.

One consequence worth internalising before implementing §4.5.3 as a
queue: the queue's WRITE side is commit-clocked and its READ side is
handshake-clocked, and those two clocks never talk to each other in
this design — commit-side logic MUST NOT inspect queue-full to decide
whether the write happened (the write HAS happened the moment commit
fires; a "queue full, write ignored, no response" state is not merely
non-compliant, it is the single most creative way this chapter knows
to violate §4.3, found independently by two designs that thought they
were protecting the response path). Fullness governs ACCEPTANCE (§4.5.3's last
clause), never EFFECTS: by the time the queue could notice anything,
the write is already the register file's problem, and the only
remaining question is which response is owed.


---

## 4.6 Read path

ARVALID/ARREADY gates address acceptance; the slave returns R with
RVALID asserted until RREADY. Reads are non-destructive except where
the register map says otherwise (ADS-8851 STAT/ERR_STAT semantics,
ES-2.1.3).

### 4.6.1 AR and R channel rules

1. The AR transfer (address accepted) is the entire request: there is
   no read data "pair" to interleave — read data flows solely on R.
2. RVALID is generated after (or coincident with) the AR transfer,
   within the latency bound of §4.8; RVALID holds until RREADY,
   RDATA/RRESP stable while held — mirrors of §4.5.1 rules with the
   R payload.
3. A read that has not been transferred (no R handshake) has had NO
   side effect. This is not obvious and is normative anyway: the
   ADS-8851 read side includes side-effecting registers whose effect
   is defined at the R TRANSFER (the edge of RVALID&RREADY), not at
   RVALID assertion and not at AR acceptance. An intermediate that
   speculatively consumes, duplicates, or answers R before its transfer
   corrupts exactly those registers (es. erratum ES-2.1.3 belongs to
   the same family of "a read is an event, not a peek").
4. The host MAY hold RREADY LOW for arbitrary wait states; the slave
   MUST keep RDATA available across them and MUST NOT re-issue or
   change the read's side-effect bookkeeping because of the wait.
   Bridges that "time out" a held RREADY and re-drive R later are
   double-firing the event in rule 3's registers; see also §4.11 item
   C7.

### 4.6.2 Read data placement and lane usage

RDATA is the full 32-bit register value, little-endian lanes, no read
strobe — reads are always full-word. Reserved and undocumented bits
read per their register-map definition (rsvd reads 0 on the ADS-8851);
there is no lane-granular read concept in Lite. The data port of the
ADS-8851 (window-result registers at 0x40..0x5C) obeys the ordinary
read rules at bus level — what makes it special is its register-level
snapshot semantics (datasheet §5/D.4), invisible on this bus.

### 4.6.3 Reads and ordering

A read's R response is ordered with respect to other reads the same
way §4.5.2 orders writes; a read is ordered after a preceding write
(one that was issued earlier and whose B has not yet transferred).
The consequence hosts rely on daily: software may read back a config
register immediately after writing it, in program order, without an
explicit barrier — the ordering rules make the readback see the write.
(On the ADS-8851 the readback MAY legitimately show a
semantically-different value because a register can be self-modifying
at window close — a content question, not an ordering one; see §4.11
item C9 and the D.4 snapshot rules it cites.)

### 4.6.4 Read data sources and capture instants *(informative)*

RDATA may be sourced from a register cell, a synchronised snapshot, or
a state machine result — the bus rules constrain the CAPTURE instant
(the R transfer edge, §4.6.1 rule 3) and the ordering (§4.6.3), not
the internal path, with two consequences this chapter states only as
constraints (the endpoint's own documents own the details):

1. If the value changes between driving it under RVALID and the
   transfer edge (a self-modifying register advancing at window
   close), the value at the transfer edge IS the read's value. A
   slave that presents one value under RVALID and a different one at
   the transfer has violated payload stability (§4.1 rule 3 binds
   RDATA exactly as it binds AWADDR) — and any test that "passes"
   against such a slave did so with held-RREADY traffic it never
   constructed.
2. Side-effect bookkeeping (DRDY consumption, read-to-clear) keys off
   that same transfer edge — not off RVALID, not off the AR transfer,
   not off an internal "read is coming" prediction. Getting this one
   cycle early or late is the difference between a destructive read
   and a missing or extra one, and the ES-2.1.3-class violation
   counters are the forensics that catch a hardware mis-sampling with
   exactly the fidelity they catch software double-reads: the register
   layer cannot tell WHO consumed, only that somebody did.

---

## 4.7 Responses and errors

- B/RESP = `OKAY` (2'b00) for legal accesses.
- `SLVERR` (2'b10) for: decode errors (reserved address, bad WSTRB),
  and write-only/read-only violations.
- `EXOKAY`, `DECERR` are not used on this interface.

### 4.7.1 Response generation table

| condition (detected at commit / at R issue) | response | effects on the target |
|---|---|---|
| legal access to implemented register, full WSTRB | OKAY | as the register defines |
| legal no-op write (WSTRB 0x0) | OKAY | none (by definition) |
| partial WSTRB to register (§4.4 platform rule) | SLVERR | MUST be none (suppressed) |
| reserved / unimplemented address (including aligned holes) | SLVERR | none |
| write to read-only register (data port, STAT, ALARM_STATUS, WAVE_*) | SLVERR | none |
| read of a write-only register | SLVERR | RDATA undefined (hosts MUST ignore), zero side effects on the target |
| access to a register locked out by lifecycle/test mode | SLVERR (ADS-8851: §6.3 class table; test-mode access rules are datasheet territory, not bus) | none |

### 4.7.2 When the code is decided

The response code is FIXED at the moment the response is generated
(commit for writes, R-issue for reads). An error detected later than
the response is issued (a late parity hit on a just-served read, a
register self-modifying after its own read transfer) does not alter
that response; it belongs to registers and interrupts, not to bus
codes. Late-detection errors routed via B/RRESP are ILLEGAL twice:
they break §4.5.1 rule 3's stability and they lie backward in time.

### 4.7.3 What SLVERR does NOT mean

SLVERR is a delivered-transaction outcome, not an exception: the
transaction fully consumed its ordering position, and (on this
endpoint) latched evidence (ADS-8851 ERR_STAT/ALARM_STATUS.COMMS per
its §6 definitions) exists beside the response — the response is
ephemeral, the register evidence is deliberate. Host drivers that
"treat SLVERR like N200" and continue are violating their own platform
contract (the AVV-3000 driver's policy: error walk, no blind retry —
the blind retry is the §4.3.2 double-landing hazard arriving with a
different paperwork).

### 4.7.4 Side-channel error propagation, stated so it stays out of the bus *(informative)*

A common review question with an equally common wrong answer: "should
a read of ERR_STAT show the errors that ERR_STAT is about to reveal,
in RRESP?" No. RRESP describes THIS transaction's bus outcome — OKAY;
the register it reads IS the error report, and it reads cleanly.
Mixing "the content reports errors" into "the access errored" breaks
§4.7.1 by inventing a response class with no codepoint, and makes
error polling indistinguishable from bus faults — the same conflation
that ends with a transport hiccup latching ALM_COMMS-class evidence
(the §6 definitions keep the layers separate on purpose: codes for
the pipe, registers for the house, interrupts for the neighbourhood,
lamp colours for the operator, §8.4's colour-is-UI rule). Each layer
has its own tests; review comments that merge two of the layers are
asking for the failure mode where the safelight turns red because a
cable was flaky.

### 4.7.5 Sequenced-error example *(informative)*

Three back-to-back writes to a device: good address, reserved
address (0x30 class), WAVE_CFG write, plus a partial-strobe
WAVE_CFG write inserted anywhere:

    W1 (CFG, 0xF)      commit OK       B1 OKAY
    W2 (0x30, 0xF)     reject          B2 SLVERR   (no register touched;
                                                    ERR/ALM_COMMS latch
                                                    per §6 definitions)
    W3 (WAVE_CFG, 0xF) commit OK       B3 OKAY
    W4 (WAVE_CFG, 0x7) reject          B4 SLVERR   (write suppressed —
                                                    WAVE_CFG unchanged
                                                    by W4; note it is
                                                    changed by W3, and
                                                    a host reading the
                                                    register after W4
                                                    sees W3's value —
                                                    correct, ordered,
                                                    both errors landed
                                                    nothing)

Four commits-eligible requests, four ordered responses, mixed codes,
one response each; the register file reflects ONLY W1+W3. The
endpoint's error counters (not this bus) record W2/W4 as transport
events — the split between "what the wire says happened" (B1..B4) and
"what the house records" (§6 latches) is the §4.7.4 layering doing
real work under traffic, and a test that conflates them will "find a
bug" where both books are individually correct.

---

## 4.8 Slave timing bounds (ADS-8851)

- Write commit latency ≤ 4 ACLK after both handshakes complete.
- B latency ≤ 6 ACLK from commit (each write, back-to-back included).
- Read data latency ≤ 4 ACLK from AR award.

### 4.8.1 Reading the bounds

These are endpoint maxima: every response earlier than the bound is
compliant, and the "back-to-back included" qualifier on B exists
because the failure shape of §4.3.1 hides best at exactly the
single-outstanding pause. Verification practice (informative): the
bus checker should run the response-latency probe at both extremes —
a slave that always answers at the bound and one that answers at
cycle 1 both pass a naive single-latency test, and the
response-retention queue shows its behaviour between them (the held-
BREADY trace of §4.11 item C6 spans both).

The counters in §4.6 of the D.5 datasheet view (window-occupancy on
the bus) count from these numbers: at 100 MHz-class ACLK, each
bound is double-digit nanoseconds against a 2 ms window; this
endpoint's fabric occupancy is therefore a rounding error and bus
contention is not a credible explanation for host-visible symptoms —
which is a load-bearing negative in the AVV-3000 field case studies
(datasheet A.6 budget note).

### 4.8.2 Worked worst case *(informative)*

One config write, end to end, worst case per bound, at 100 MHz ACLK
(10 ns), against a 2 ms window:

| segment | bound | ns |
|---|---|---|
| commit after second request handshake | ≤ 4 ACLK | ≤ 40 |
| B issue after commit | ≤ 6 ACLK | ≤ 60 |
| host consumes B (ready immediately) | 1 ACLK | 10 |
| **endpoint contribution to the write round-trip** | | **≤ ~110** |

Window budget: 2 000 000 ns; the configuration path runs at 0.006 %
duty before any pipelining. Same arithmetic for a data-port burst
(8 reads, pipelined, ordered per §4.6.3): comfortably under 1 µs.
This table, not intuition about buses being slow, is what governs the
credibility of "bus timing" as an explanation for window-level
symptoms — and it is quoted in the AVV-3000 field analyses whenever a
telemetry-sounding explanation tries to route through the fabric.

### 4.8.3 Where these numbers came from *(informative)*

The three §4.8 numbers are design budgets translated into maxima, not
measured surprises: the commit bound (4) is the register-file write
port's worst-case arbitration-adjacent depth (decode + latch + the arbitration stall
of §4.9 accounted SEPARATELY); the B bound (6) adds the two-stage
response queue (§4.5.6: a pass-through stage plus the holding stage —
the extra two cycles over commit is queue transit at zero congestion,
which the back-to-back case must still make); the read bound (4) is
the same register-port path with the response queue bypassed (R has
one stage — the R path never needs more depth than the §4.10 read
envelope, so its latency is a wire problem, not a queue problem).
Translating back: none of the bounds has slack the endpoint is
counting on, and none has the arbitration stall inside it — §4.9's
2 ACLK is additive headroom the design carries but hosts never budget.

---

## 4.9 Register-window arbitration note

AXI accesses arriving during the SDR engine's window-commit instant
are arbitrated in favor of the bus; commit stalls ≤ 2 ACLK (ADS-8851
§7.4.2 H-3). No state is lost; the window result is held and presented
once the bus is free.

### 4.9.1 What the arbitration cannot be

The arbitration rule is one-directional and the direction has teeth:
the fabric side has no priority mechanism against the internal
sequencer for this endpoint; "the window won't accept a write this
instant" is expressed as READY held LOW (legitimate wait), never as
SLVERR "try again" — a transient-busy BRESP response code does not
exist on this interface and inventing it in the response payload's
reserved space is a §4.7 violation. Likewise the stall bound (≤ 2
ACLK) is a bound on the internal stall, not on the host-perceivable
latency — hosts budget §4.8's numbers and see the sum; they MUST NOT
be told about the internal event at all.

For the AVV-3000's read-side analogue: a read whose side effect
(DRDY-clear in the STAT family, ES-2.1.3) coincides with a window
commit resolves with the access order the host issued — the bus
transfer order defines the event order; the internal arbitration
neither invents nor reorders events, only slides their internal commit
by ≤ 2 clocks. Section J's event-consistency patterns apply.
*(Informative; the normative behavior lives in the ADS-8851 §7.4.2
H-notes and their tests, not on this bus.)*

### 4.9.2 Worked arbitration trace *(informative)*

A write to WAVE_CFG whose commit coincides with the sequencer closing
a window:

    t0: AW and W both transferred (commit fires); the sequencer's
        window-close logic touches shared write-port state
    t1: arbitration: bus wins (no sequencer priority mechanism exists
        against the fabric for this endpoint); the sequencer observes
        an internal stall; the register file applies the bus write
    t2: sequencer resumes within the ≤ 2 ACLK stall bound; window
        result contents unaffected — the sequencer datapath and the
        bus write targets are disjoint by construction; arbitration
        exists for shared write-port contention only
    t3 (≤ t0 + §4.8 bound): B issued — the arbitration stall lives
        entirely on the internal side of the latency decomposition,
        never charged against the B bound

Non-compliance, for recognition in review: sequencer-side effects from
the stall (a skipped window close, a double-counted debounce
increment), B beyond the bound (stall charged to the wrong segment),
or a window result register presenting a half-updated value to a read
racing the stall — that last hole is closed by the data port's own
commit rules (§5, D.4), which is why this note is cross-cited from
both documents and tested from both sides.

---

## 4.10 Outstanding transactions and ordering envelope

There is no transaction ID and no explicit ordering control on
Lite — ordering is therefore the load-bearing simplification of the
protocol, and it is stated as rules:

1. **Writes commit in issue order.** "Issue order" is the order of
   their (AW, W) completion pairs; §4.2.1's FIFO pairing makes this
   unambiguous. A host MUST NOT depend on writes committing out of
   order (none can), and a slave MUST NOT let them commit out of order
   (a reordered commit is a corrupted register stream).
2. **Reads return in AR issue order.** Mirrors rule 1 on the read
   channel; same proof obligations.
3. **Read-after-write of any address is ordered by issue.** A read
   issued after a write sees that write's effects (subject to the
   register's own semantics — §4.6.3's self-modifying caveat). A read
   issued before a write MUST NOT see it, and there is no legal
   implementation in which it does.
4. **Outstanding envelope (AVV-3000 host-side policy, normative for
   hosts against this endpoint):** at most two writes issued with
   responses pending, of which at most one is "in flight" — the second
   may be AW/W-complete-but-B-pending only in the driver's documented
   write-combo sequences; at most two reads outstanding with R pending.
   §4.5.4 derives the endpoint's queue sizing from this envelope; the
   compliance suite drives beyond it to prove the envelope is
   protected by back-pressure (§4.5.3), not trusted.
5. **No cross-channel ordering beyond write-vs-read issue order.**
   An AW and an AR issued in the same cycle have no defined priority
   against each other on multi-layer interconnects; the envelope of
   rule 4 exists so the AVV-3000 driver never needs that undefined
   precedence: config-then-read sequences are separated at issue.

**The exception list.** Not-ordered by anything, ever: (a) accesses to
different endpoints behind the same host port (no global bus-wide
order exists; the D.5 view is one endpoint), (b) accesses whose B
response transferred at different times on different slaves
(rule (a) again), and (c) the internal sequencer's own events, which
this bus does not see (§4.9).

### 4.10.6 Timeouts against a silent endpoint *(informative)*

Nothing in this chapter requires the endpoint to answer within
anything but the §4.8 bounds AFTER acceptance; readiness starvation (a
READY held forever, §4.1 rule 5 territory taken to its failure
outcome) is not a protocol event — the transaction simply has not
happened, and any "response timeout" lives entirely in the host. Host
policy on the AVV-3000 stack: (1) a pending transaction past the
envelope-derived bound is an ERROR CONDITION, logged, never silently
retried (retry without log is §4.3.2's amplification path arriving
via the host side); (2) the recovery from a stalled endpoint is
RESET and re-init (§4.1.5), not a repeated write to the register that
"isn't answering" — a write that isn't being accepted cannot be made
more acceptable by sending its twin; (3) timeouts during clinical
flow are safety events, not retries — the pump's response to "the
sensor ASIC has stopped answering my writes" belongs to the system
safety spec, and its answer is never "keep trying until the watchdog".

### 4.10.7 Why Lite has no reorder/ID machinery — a rationale *(informative)*

Full AXI4 buys concurrency with IDs and reorder buffers and pays for
it in verification surface everywhere downstream — including,
notoriously, in every device that must decide what "ordering of
transaction X vs Y" means for its register semantics. Lite trades the
bandwidth for the property this chapter leans on absolutely: the
transaction order IS the issue order, unambiguously, with no
configuration. On an embedded control plane (one host, a handful of
endpoints, 2 ms sensor windows that the §4.8.2 table already leaves
five orders of magnitude away from the bus's duty cycle) there is no
traffic pattern the ordering simplification costs anything against —
every reordering mechanism that "would have helped" here is a
compliance liability, including the creative ones (response merges —
§4.5.6; speculative reads — §4.6.1; response-side error re-decisions
— §4.7.2). The chapter is short because the protocol removed the
questions, not because it forgot them.

---

## 4.11 Compliance checklist for bridge implementers *(informative)*

The rules above, as a test list — every item is a trace the suite can
construct; every item has been the subject of a real non-compliance
report at least once in the family. An implementation passes §4 by
demonstrating all of these, including the negative tests.

| # | construct | pass condition |
|---|---|---|
| C1 | single write, host holds BREADY k cycles | BVALID stable, BRESP stable, transfer at k; commit effects landed ONCE |
| C2 | two writes back-to-back, B1 unconsumed at #2's commit | exactly two responses, in order, per §4.3 (THE t_bus case) |
| C3 | sustained write stream with random BREADY gaps > inter-write spacing | response count == write count always; no merge, no loss, no dup |
| C4 | WSTRB sweep 0x0, 0xF, partials | OKAY/no-op, ordinary, SLVERR/suppressed (register byte-identical before/after) |
| C5 | decode-error writes; write-to-read-only; read-of-write-only | SLVERR, zero side effects, ordering position intact (§4.7.1) |
| C6 | response-latency sweep from 1 ACLK to bound + over-bound fault injection | within bound: always OKAY in order; over-bound: NOT an endpoint behaviour — the test verifies the checkers fire, the design never satisfies this row by luck |
| C7 | R side: RREADY held during RVALID; a second AR queued behind; side-effect register reads | side effect exactly once at transfer; second read of side-effect register returns post-effect value; no timeout-driven re-issue (rule §4.6.1-3/4) |
| C8 | write burst to the envelope (+1 beyond it) | AWREADY/WREADY back-pressure; zero drops/merges; after drain, everything in order (§4.5.3, §4.10-4) |
| C9 | write→same-address read loops against self-modifying registers (STAT/ERR_STAT families) | readback consistent with ISSUE order; per-register semantics (destructive/RC/snapshot) per their datasheet definitions — this row checks the bus does not scramble events, not that registers are plain |
| C10 | interleave: read of STAT during held-B of a wave-config write (envelope-permitted overlap) | both transactions complete with their own semantics; no cross-bleed of side effects (write committed once, read side-effected once) |
| C11 | reset/teardown mid-transaction | no response emitted for never-committed writes; committed-then-reset keeps §4.3 (the response obligation is committed-state, but note it may be lost to reset ONLY for endpoints whose map says reset clears it — ADS-8851: pending B responses are NOT guaranteed across SWRST; this is the documented exception and hosts MUST time out rather than hang) |

The negative rows (C4-partial, C8-beyond-envelope, C6-over-bound)
matter as much as the positive ones: a bridge that never sees them
because its environment never constructs them is untested in its
error-handling half, which is the half with the field tickets.

### 4.11.1 Harness note — sampling and the one-cycle windows *(informative)*

Every one-cycle window in this chapter — the AR/R coincident-accept
edge, the back-to-back B raise, the RVALID&RREADY transfer edge that
fires the side effect — is one *edge* wide, and verification harnesses
that sample state AFTER advancing the clock miss them. Established
practice on this codebase (learned the usual way once, then written
down): the bus driver captures handshake state BEFORE the tick that
transfers it, and compliance checks read those captured values —
post-tick checks see one-cycle signals already gone and produce
"impossible" failures at exactly the rates legal traffic creates. A
checker that cannot observe a legal event the protocol requires is
observing the wrong cycle; fix the sampling point before believing the
design.

---

## 4.12 Signal reference (AVV-3000 instantiation) *(informative — full normative signal list is AM004 ch.2)*

    SIGNAL          WIDTH DIR(src -> dst)     CHANNEL  NOTES
    s_axi_awaddr    32    host -> endpoint    AW       byte address, word aligned in practice
    s_axi_awprot    3     host -> endpoint    AW       carried; endpoint policy = ignore (unprivileged trust domain, datasheet §4)
    s_axi_awvalid   1     host -> endpoint    AW
    s_axi_awready   1     endpoint -> host    AW
    s_axi_wdata     32    host -> endpoint    W
    s_axi_wstrb     4     host -> endpoint    W        full-word semantics: §4.4
    s_axi_wvalid    1     host -> endpoint    W
    s_axi_wready    1     endpoint -> host    W
    s_axi_bresp     2     endpoint -> host    B        OKAY/SLVERR only on this endpoint
    s_axi_bvalid    1     endpoint -> host    B
    s_axi_bready    1     host -> endpoint    B
    s_axi_araddr    32    host -> endpoint    AR       word aligned
    s_axi_arprot    3     host -> endpoint    AR       ignored as awprot
    s_axi_arvalid   1     host -> endpoint    AR
    s_axi_arready   1     endpoint -> host    AR
    s_axi_rdata     32    endpoint -> host    R
    s_axi_rresp     2     endpoint -> host    R        OKAY/SLVERR only
    s_axi_rvalid    1     endpoint -> host    R
    s_axi_rready    1     host -> endpoint    R

Naming here follows the endpoint-side convention (`s_axi_*`, slave
view); the host-side mirror in the fabric prefixes `m_axi_*`. No
optional Lite signals (AwLOCK, AwCACHE, AwQOS, AwREGION, user lanes)
are connected at this endpoint; absent means absent, not zero — a
bridge injecting them into the endpoint's view changes nothing about
the rules of this chapter.

### 4.12.1 Port census *(informative)*

Full endpoint surface: 6 request-channel signals + 3 address/prot +
5 data/write + 3 B + 4 R-side, 32 total wires; no optional signals
implemented. Direction census from the endpoint view: 15 inputs
(addresses, controls from the host), 15 outputs (readies, responses,
data), 2 clocks/resets in the AVV-3000 instantiation (ACLK, ARESETn;
the CDC note for the datasheet's §6.5 is not this chapter's subject —
the bus rules apply at the ACLK domain edge and everything in chapter
4 speaks that clock exclusively).


---

## 4.13 Conventions, terms, and how to read this extract

- **Endpoint / slave** are used interchangeably (AM004 ch.2 says
  subordinate; the AVV-3000 tree says bridge/slave; the chapter text
  means the RESPONDING side of the request channels).
- **Transfer / transaction / access**: transfer = one channel handshake
  edge; transaction = one write (AW+W+B) or read (AR+R) lifecycle;
  access = one transaction at the register file. "Commit" is defined
  in §4.2; "response" never means anything except a B or R handshake
  that actually transferred.
- **Non-destructive read** here means at the BUS interface level.
  Side-effecting registers (STAT-family, RC registers) are
  destructive/collecting AT THE REGISTER LEVEL by design — the
  statement in §4.6's lead is about what reads cannot do (invent,
  duplicate, reorder), and register semantics decide what reads MUST
  do. Both statements coexist; the ES-2.1.3 confusion is reading the
  bus sentence into the register file.
- Extract editorial marks: *informative* sections are this extract's
  additions to AM004's body text and cannot be cited as normative for
  compliance runs beyond AM004's chapters — they exist so the
  normative sentences have worked companions. The normative
  material cited from ADS-8851 documents (§4.3 invariant, §4.5.3
  retention, §4.4 WSTRB platform rule, §4.8 bounds) is carried in
  body text and IS citable.

*Chapter boundary note — full AM004 chapter 4 continues with channel
timing diagrams (4.x figures; not carried in this extract — the
handshake figures' ASCII equivalents appear in §4.1.2 and §4.3.1 of
this extract) and the protocol-performance monitor hooks (not used on
AVV-3000). The ADS-8851 datasheet's internal view of the same
mechanisms carries the implementation narrative: §4 for the bus
surface, D.5 for the slave internals, §7.4.2 for the read-protocol
contract built on top of these rules.*

---

## 4.14 Compliance case study: the back-to-back pair, cycle by cycle *(informative — the C2 trace in prose)*

The full lifecycle of the C2 construct (two writes, first response
deliberately unreceived), stated per §4 edge so a reader can diff it
against a waveform dump:

    edge  event
    t0    AWREADY high, AWVALID high     -> AW transfer W1 (addr #1 taken)
    t1    WREADY high, WVALID high       -> W  transfer W1  == COMMIT #1
    t1    (same edge) queue push: B1 = OKAY; BVALID raising as of t1
    t2..  host holds BREADY LOW; BVALID stays; BRESP stays
    t3    AW transfer W2 (AWREADY high again — the §4.10 envelope's
          second slot; AWREADY is NOT gated on B1 — request progress
          and response consumption are decoupled by design)
    t5    W transfer W2 == COMMIT #2; queue push: B2 behind B1
          (BVALID already high offering B1: the queue does not "raise
          again", the offering continues — §4.5.1)
    t6..  still no BREADY; both commits are facts; the register file
          has both writes landed, in order, and the wire owes B1, B2
    t12   host raises BREADY -> B1 transfers (edge of BVALID & BREADY)
    t13   queue non-empty -> BVALID CONTINUES (no drop cycle, no
          re-raise ceremony): B2 is offered from t13
    t14   BREADY high -> B2 transfers. The trace closes with exactly
          two responses for two commits.

The non-compliant variant differs at exactly one edge — the one where
commit #2 overwrites the pending-B1 slot — and nothing observable at
any other edge reveals it. A verification plan that spot-checks
"responses eventually arrive" on random traffic has a probability, not
a proof, of constructing the t3-while-t2 window, which is the entire
specification reason the C2 row of the checklist is mandatory and
fixed-seed rather than coveraged.

## 4.15 Terms used in this chapter's body *(informative index)*

- **AW/W pairing rule** — first address pairs with first data; §4.2.1.
- **back-pressure** — refusing acceptance (READY LOW) as the ONLY
  legal congestion response on this interface; §4.5.3, §4.9.
- **bijection (response/commit)** — P2; one response per commit and
  vice versa; §4.3.1-notes.
- **commit** — the moment a write is irrevocable and its response is
  owed; §4.2; distinct from acceptance and from response transfer.
- **destructive read** — register-level side effect at the R transfer
  edge; §4.6.1 rule 3; a REGISTER property (ES-2.1.3), not a bus
  response class (§4.7.4).
- **envelope** — the host-side outstanding limits the endpoint's
  response-retention depth derives from; §4.5.4, §4.10 rule 4.
- **held BREADY / held RREADY** — the host exercising its wait-state
  right; the stress corner every retention rule exists for.
- **issue-order equivalence** — commit order == host issue order ==
  response order; §4.5.2, §4.10 rules 1–3.
- **retained response** — a pending B the endpoint MUST keep offering
  until its transfer; §4.5.3.
- **transfer** — the handshake edge; the only event the rest of the
  chapter is written about.

## 4.16 Adjacent specifications *(informative pointers)*

This extract never stands alone for a full implementation; what it
does NOT own: register semantics and access classes (ADS-8851 §6 —
including which reads destroy and which writes latch), the init and
recovery sequences that sit ABOVE bus transactions (ADS-8851 §7), the
interrupt/aggregation behaviour built on the latches responses
describe (§6.12), the data-port snapshot rules that make reads of
0x40..0x5C well-defined against the sequencer (§5/D.4), and the
fabric-side view above this bridge (D.5). The chapter boundary is
deliberately transactional: everything here is about a wire that a
logic analyzer can watch; everything that needs a datasheet to
interpret the VALUES on that wire lives in the datasheet. A
reviewer's short version: if the question is "may the bus do this" →
this chapter; if it is "what does the device now know" → ADS-8851;
if it is both → it is an ordering question, and ordering questions
are sections 4.2.1, 4.5.2, 4.6.3, and 4.10, in that reading order.

## 4.17 Legacy host stacks on a compliant endpoint *(informative)*

A final category the compliance suite carries because the field
carries it: bus traffic that is perfectly legal AXI and still wrong
for this endpoint. None of it is catchable on the wire description of
sections 4.1–4.10 — all of it IS catchable by the endpoint's own
registers, which is the load-bearing fact of this section:

- **Redundant reads.** A master that reads STATUS, reads it again a
  cycle later "to be sure", and proceeds on the second value is doing
  five legal AR/R transactions where two suffice, and on the
  ADS-8851 it is converting legal behaviour into register violations —
  the transfer edge of §4.6.1 rule 3 fires the first read's effect,
  and the second read's existence is itself the event ERRSTAT counts.
  The bus layer CANNOT see that this is the bug: identical waveforms
  appear for correct and incorrect drivers. Only the register layer
  (and the errata that documents its semantics) says otherwise.
- **Read-modify-write races dressed as ordering.** Two hosts (or two
  threads through the same bridge) doing RMW against the same
  config register produce, at the wire, legal accesses in whatever
  order arbitration served them; §4.10's rules describe the
  orderings' LEGALITY, not their SAFETY. Lite has no atomic compare
  and this chapter invents none; exclusive flows are a full-AXI
  feature and the endpoint implements no version of them (§4.5.5).
  Systems with two writers own software arbitration (or a hardware
  lock at the fabric level that this endpoint, correctly, knows
  nothing about).
- **"Optimistic" write combining.** A host bridge coalescing two
  writes to the same register into one — every Lite transaction owes
  exactly one response (§4.3), so the combining bridge either answers
  the second write for the device (fabric-side fiction: two commits
  never existed on this wire, responses are locally manufactured —
  LEGAL only for pure-memory behind-buffers and ILLEGAL for
  transactional registers, which the ADS-8851's map is entirely made
  of) or it is not combining at all. Any fabric feature phrased as
  "write merging", "write absorption", or "same-address coalescing"
  is disabled at review for this endpoint's window, full stop.

The common shape of the whole section: the bus is the part of the
system with the fewest degrees of freedom, and legacy behaviour that
predates endpoint semantics survives bus compliance perfectly. The
verification that protects AVV-3000 deployments is therefore paired:
this chapter's negative tests on the wire, and the endpoint's
register-side counters (violation latches, error classes per §6) in
the acceptance suite — a driver change that passes C1..C11 but starts
banking ERR_STAT entries has NOT passed; it has merely moved the bug
from the analyzer to the ASIC, where it is being tallied for the
next person's field report.

## 4.18 Review case reviews *(informative — anonymised, all real shapes)*

Three §4-class findings as they appeared in bridge reviews on this and
adjacent platforms. Each was merged-then-caught; each maps to a
checklist row so the next design's reviewer starts from the row, not
the scar.

**Case A — "the state machine that cleaned up."** A bridge added an
idle-timeout that cancelled BVALID after 256 unaccepted cycles to
"prevent hangs". The endpoint could indeed no longer hang the fabric —
because it also no longer owed any response, converting every slow-
BREADY host into the §4.3.1 corruption case at exactly the rate the
timeout fired, and the field symptoms (values landing twice) appeared
ONLY at the 37 °C clock-throttle corner where BREADY waited past 256
cycles. Checklist rows: C1, C3, and §4.5.1 rule 2 have a 256-
infinity-free clause for this reason — the response obligation has no
timer anywhere in this specification, and the only legal way for a
transaction to stop existing is to have never committed.

**Case B — the merged config push.** A board-support writer,
optimising register init, coalesced repeated writes to the same
address in a loop ("same value, twice, surely idempotent") into one
transaction. The init walk happened to include a WAVE_CFG write whose
second instance was the RE-ARM edge (write, mask, re-write), and the
coalescer was correct that the DATA were identical and wrong that the
transactions were: arm and re-arm are the same bytes and different
events (§4.3's bijection is per transaction, and so is the hardware's
reaction). The failure signature — a waveform profile starting late
only on the "fast" boot path — cost two debug days and produced
checklist case C2's second fixed-seed instance on the host side.
Idempotence is a per-register property (and §6's map owns it), never
a per-transaction permission.

**Case C — reset overlap.** A warm restart issued while a write was in
flight "lost" the write's effects across the reset and was filed as a
response-loss defect. Correct behaviour per §4.1.5 and §4.11 row C11:
the endpoint's pending response died WITH the state that was being
responded about, and re-initialisation (the §7.2 walk) was already
underway. The real defect was the host assuming "the write returned"
in a log line that printed on AW/W acceptance — the §4.2.3 warning in
the wild, with the additional lesson that reset-boundary traces need
the endpoint's init walk in the same capture window or they
mis-prosecute compliant hardware.

## 4.19 Half-formed transactions *(informative)*

For completeness, the states between reset edges and protocol edges,
since verification writes coverage targets against them:

- **AW accepted, never paired** (host died between AW and W): the
  endpoint holds a request-side queue entry forever (or until reset —
  the envelope of §4.10 means this is bounded, not zero; a bridge
  whose AW FIFO cannot drain because W never comes is correct
  behavior with a dead master and MUST NOT "self-heal" by pairing it
  with the next live write's datum — that is §4.2.1's corruption in
  its least likely, ugliest form).
- **Committed, response never consumed, host gone:** same obligation,
  forever; the endpoint is allowed to be stuck behind the offering
  (it has satisfied every rule by continuing to offer — a bus with no
  takers has no requirements), and no timeout anywhere in the standard
  releases the obligation (§4.1.5's reset clause is the ONLY
  eraser).
- **RVALID offered, host vanished between edges:** the read's side
  effect has NOT fired (no edge); the next reader gets the un-
  consumed event. Verification pair: a read abandoned at RVALID−1 and
  a read issued later must together show exactly one consumption —
  the counterpart test of C7 that no one writes until the first
  STAT-family bug report arrives.

These three shapes share one spec sentence: **protocol edges are the
only facts about state changes**, and §4's whole apparatus — stability
rules, transfer edges, offer obligations — is machinery for naming
them. Coverage targets that skip the half-states assert the happy
projection of that sentence, which is exactly where "it passed
directed tests and failed in the field" comes from.
