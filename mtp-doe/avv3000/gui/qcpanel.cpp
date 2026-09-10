// qcpanel.cpp — see qcpanel.h. No arithmetic here beyond QString::number.
#include "qcpanel.h"

#include <QLabel>
#include <QVBoxLayout>

QcPanel::QcPanel(QWidget *parent) : QWidget(parent)
{
    qc_rate_reset(&m_rate_st);
    auto *lay = new QVBoxLayout(this);
    m_rate = new QLabel(this);
    m_rate->setObjectName(QStringLiteral("rateLabel"));
    lay->addWidget(m_rate);
    setWindowTitle(QStringLiteral("AVV-3000 QC Panel"));
    render();
}

void QcPanel::feedSample(uint32_t ts_ms, uint32_t counts)
{
    qc_rate_update(&m_rate_st, ts_ms, counts);
    render();
}

QString QcPanel::rateText() const
{
    return m_rate->text();
}

void QcPanel::render()
{
    if (m_rate_st.cps < 0) {
        m_rate->setText(QStringLiteral("rate: --"));
        return;
    }
    // counts/sec straight from the core; NO further scaling here
    m_rate->setText(QStringLiteral("rate: ")
                    + QString::number(m_rate_st.cps)
                    + QStringLiteral(" counts/s"));
}
