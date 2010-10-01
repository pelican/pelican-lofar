#include "TimeSeriesDataSet.h"

#include <QtCore/QIODevice>
#include <QtCore/QTextStream>
#include <QtCore/QFile>
#include <QtCore/QString>

#include <iostream>

namespace pelican {
namespace lofar {

void TimeSeriesDataSetC32::write(const QString& fileName,
        int s, int p, int b) const
{
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return;

    const std::complex<float> * times = 0;

    unsigned sStart = (s == -1) ? 0 : s;
    unsigned sEnd = (s == -1) ? nSubbands() : s + 1;
    unsigned pStart = (p == -1) ? 0 : p;
    unsigned pEnd = (p == -1) ? nPolarisations() : p + 1;
    unsigned bStart = (b == -1) ? 0 : b;
    unsigned bEnd = (b == -1) ? nTimeBlocks() : b + 1;

    QTextStream out(&file);
    for (unsigned s = sStart; s < sEnd; ++s) {
        for (unsigned p = pStart; p < pEnd; ++p) {
            for (unsigned b = bStart; b < bEnd; ++b) {

                // Get the pointer to the time series.
                times = timeSeriesData(b, s, p);

                for (unsigned t = 0; t < nTimesPerBlock(); ++t)
                {
                    out << QString::number(times[t].real(), 'g', 8) << " ";
                    out << QString::number(times[t].imag(), 'g', 8);
                    out << endl;
                }
                //out << endl;
            }
            out << endl;
        }
        out << endl;
    }
    file.close();
}


}// namespace lofar
}// namespace pelican

