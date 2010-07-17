#include "SubbandTimeSeries.h"

#include <QtCore/QIODevice>
#include <QtCore/QTextStream>
#include <QtCore/QFile>
#include <QtCore/QString>

#include <iostream>

namespace pelican {
namespace lofar {

void SubbandTimeSeriesC32::write(const QString& fileName) const
{
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return;
    }

    QTextStream out(&file);
    for (unsigned index = 0, b = 0; b < _nTimeBlocks; ++b) {
        for (unsigned s = 0; s < _nSubbands; ++s) {
            for (unsigned p = 0; p < _nPolarisations; ++p) {

                // Get a pointer the the spectrum.
                const std::complex<float>* times = _data[index].ptr();
                unsigned nTimes = _data[index].nTimes();

                for (unsigned t = 0; t < nTimes; ++t) {
                    double re = times[t].real();
                    double im = times[t].imag();
                    out << QString::number(re, 'g', 16) << " ";
                    out << QString::number(im, 'g', 16) << endl;
                }
                index++;
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

