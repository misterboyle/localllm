// t_gui_unit.cpp — r_gui (5 pts): QC panel unit-scale contract, AN-8851-07
// §5. Feed synthetic count sequences, assert the DISPLAYED value equals
// true counts/sec. Golden passes; B5 (conversion applied twice — the
// field symptom "rate spike x1000 times nominal" — or dropped) fails.
//
// Runs offscreen: `QT_QPA_PLATFORM=offscreen` (set by gui/Makefile) —
// no window system required, hermetic like every other rung.
#include <QtTest>

extern "C" {
#include "qc_rate.h"
}
#include "qcpanel.h"

class TestGuiUnit : public QObject
{
    Q_OBJECT

private slots:
    void coreExactRate();
    void coreTruncatesOnce();
    void coreChainAndWrap();
    void coreClampAndDegenerate();
    void panelDisplaysCountsPerSec();
    void panelTracksUpdates();
};

// 10 counts over 4 ms = 2.5 counts/ms = 2500 counts/s. A double ×1000
// reads 2 500 000; a missing conversion reads 2 — both fail below.
void TestGuiUnit::coreExactRate()
{
    qc_rate_t st;
    qc_rate_reset(&st);
    QCOMPARE(qc_rate_update(&st, 0, 100), (int32_t)-1);   // baseline
    QCOMPARE(qc_rate_update(&st, 4, 110), (int32_t)2500);
}

void TestGuiUnit::coreTruncatesOnce()
{
    qc_rate_t st;
    qc_rate_reset(&st);
    qc_rate_update(&st, 100, 0);
    QCOMPARE(qc_rate_update(&st, 103, 1), (int32_t)333);  // 1000/3
    QCOMPARE(qc_rate_update(&st, 106, 2), (int32_t)333);
}

void TestGuiUnit::coreChainAndWrap()
{
    qc_rate_t st;
    qc_rate_reset(&st);
    qc_rate_update(&st, 0, 0);
    qc_rate_update(&st, 10, 25);
    QCOMPARE(qc_rate_update(&st, 20, 50), (int32_t)2500); // steady stream
    // uint32 count wrap: 0xFFFF FFF0 -> 0x0000 0010 = 32 counts / 8 ms
    qc_rate_reset(&st);
    qc_rate_update(&st, 1000, 0xFFFFFFF0u);
    QCOMPARE(qc_rate_update(&st, 1008, 0x00000010u), (int32_t)4000);
}

void TestGuiUnit::coreClampAndDegenerate()
{
    qc_rate_t st;
    qc_rate_reset(&st);
    qc_rate_update(&st, 0, 0);
    QCOMPARE(qc_rate_update(&st, 1, UINT32_MAX), (int32_t)INT32_MAX); // clamp
    // same timestamp: no interval, previous value kept
    qc_rate_reset(&st);
    qc_rate_update(&st, 5, 7);
    QCOMPARE(qc_rate_update(&st, 5, 9999), (int32_t)-1);
}

// The panel must render exactly what the core produced — presentation-only
// (a re-scale here is the B5a/b class of bug and moves this string ×1000).
void TestGuiUnit::panelDisplaysCountsPerSec()
{
    QcPanel panel;
    QCOMPARE(panel.windowTitle(), QString("AVV-3000 QC Panel"));
    QCOMPARE(panel.rateText(), QString("rate: --"));
    panel.feedSample(0, 100);                    // baseline only
    QCOMPARE(panel.rateText(), QString("rate: --"));
    panel.feedSample(4, 110);                    // 2500 counts/s
    QCOMPARE(panel.rateText(), QString("rate: 2500 counts/s"));
}

void TestGuiUnit::panelTracksUpdates()
{
    QcPanel panel;
    panel.feedSample(0, 1000);
    panel.feedSample(2, 1003);                   // 3/2 ms -> 1500
    QCOMPARE(panel.rateText(), QString("rate: 1500 counts/s"));
    panel.feedSample(4, 1003);                   // pause: rate 0
    QCOMPARE(panel.rateText(), QString("rate: 0 counts/s"));
}

QTEST_MAIN(TestGuiUnit)
#include "t_gui_unit.moc"
