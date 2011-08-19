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
        int subband, int pol, int block) const
{
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return;

    const std::complex<float> * times = 0;

    unsigned subband_start = (subband == -1) ? 0 : subband;
    unsigned subband_end   = (subband == -1) ? nSubbands() : subband + 1;
    unsigned pol_start     = (pol == -1)     ? 0 : pol;
    unsigned pol_end       = (pol == -1)     ? nPolarisations() : pol + 1;
    unsigned block_start   = (block == -1)   ? 0 : block;
    unsigned block_end     = (block == -1)   ? nTimeBlocks() : block + 1;

    QTextStream out(&file);
    for (unsigned s = subband_start; s < subband_end; ++s)
    {
        for (unsigned p = pol_start; p < pol_end; ++p)
        {
            for (unsigned b = block_start; b < block_end; ++b)
            {
                // Get the pointer to the time series.
                times = timeSeriesData(b, s, p);

                for (unsigned t = 0; t < nTimesPerBlock(); ++t)
                {
                    out << b * nTimesPerBlock() + t << " ";
                    out << QString::number(times[t].real(), 'g', 6) << " ";
                    out << QString::number(times[t].imag(), 'g', 6);
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

