// qcpanel.h — AVV-3000 QC panel (rung r_gui). Presentation skin over the
// qc_rate core: the ONLY unit-scale transformation is qc_rate's (once);
// the panel renders what the core returns. Offscreen-safe (no show()).
//
// B5 (field symptom "rate spike x1000") lands on paths that break this
// separation — re-scaling here or dropping the core conversion — and
// t_gui_unit asserts the rendered text against true counts/sec.
#ifndef QCPANEL_H
#define QCPANEL_H

#include <QString>
#include <QWidget>

class QLabel;

extern "C" {
#include "qc_rate.h"
}

class QcPanel : public QWidget
{
public:
    explicit QcPanel(QWidget *parent = nullptr);

    // Feed engine sample (cumulative counts at ts_ms); refreshes the label.
    void feedSample(uint32_t ts_ms, uint32_t counts);
    // (windowTitle() from QWidget verifies the offscreen shell exists)

    // current rate line, exactly as displayed (pure readback, no re-scale)
    QString rateText() const;

private:
    void render();
    QLabel *m_rate;
    qc_rate_t m_rate_st;
};

#endif // QCPANEL_H
