#include "TimeSeriesDataSet.h"

#include <QtCore/QIODevice>
#include <QtCore/QTextStream>
#include <QtCore/QFile>
#include <QtCore/QDataStream>
#include <QtCore/QString>

#include <iostream>

namespace pelican {
namespace lofar {

void TimeSeriesDataSetC32::serialise(QIODevice& device) const
{
     QDataStream out(&device);
     out.setVersion(QDataStream::Qt_4_0);
     out << nSubbands();
     out << nPolarisations();
     out << nTimesPerBlock();
     out << nTimeBlocks();
     out << _lofarTimestamp;
     out << _blockRate;
     for(unsigned int i=0; i < _data.size(); ++i ) {
        out << _data[i].real();
        out << _data[i].imag();
     }
}

void TimeSeriesDataSetC32::deserialise(QIODevice& device, QSysInfo::Endian) {
     QDataStream in(&device);
     in.setVersion(QDataStream::Qt_4_0);
     unsigned nSubbands, nPolarisations, nTimesPerBlock, nTimeBlocks;
     in >> nSubbands;
     in >> nPolarisations;
     in >> nTimesPerBlock;
     in >> nTimeBlocks;
     in >> _lofarTimestamp;
     in >> _blockRate;
     resize(nTimeBlocks,nSubbands,nPolarisations,nTimesPerBlock);
     long size = nSubbands * nPolarisations * nTimeBlocks * nTimesPerBlock;
     float real, imag;
     for( int i=0; i < size; ++i ) {
         in >> real;
         in >> imag; 
        _data[i] = std::complex<float>(real,imag);
     }
}

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

