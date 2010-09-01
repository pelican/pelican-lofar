#include "TimeSeriesDataSet.h"

#include <QtCore/QIODevice>
#include <QtCore/QTextStream>
#include <QtCore/QFile>
#include <QtCore/QString>

#include <iostream>

namespace pelican {
namespace lofar {

void TimeSeriesDataSetC32::write(const QString& fileName) const
{
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return;

    const std::complex<float> * times = 0;

    QTextStream out(&file);

    for (unsigned b = 0; b < nTimeBlocks(); ++b) {
        for (unsigned s = 0; s < nSubbands(); ++s) {
            for (unsigned p = 0; p < nPolarisations(); ++p) {
                times = timeSeriesData(b, s, p);
                for (unsigned t = 0; t < nTimesPerBlock(); ++t) {
                    out << QString::number(times[t].real(), 'g', 16) << " ";
                    out << QString::number(times[t].imag(), 'g', 16);
                    out << endl;
                }
                out << endl;
            }
            out << endl;
        }
        out << endl;
    }
    file.close();
}


}// namespace lofar
}// namespace pelican

