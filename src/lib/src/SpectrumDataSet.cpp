#include "SpectrumDataSet.h"

#include <QtCore/QIODevice>
#include <QtCore/QTextStream>
#include <QtCore/QFile>
#include <QtCore/QString>

#include <iostream>

namespace pelican {
namespace lofar {

/**
 * @details
 * Writes the contents of the subband spectra to an ASCII file.
 *
 * @param fileName The name of the file to write to.
 */
void SpectrumDataSetC32::write(const QString& fileName, int s, int p, int b) const
{
    QFile file(fileName);
    if (QFile::exists(fileName)) QFile::remove(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return;

    const std::complex<float>* data = 0;

    unsigned nChan = nChannels();
    unsigned bStart = (b == -1) ? 0 : b;
    unsigned bEnd = (b == -1) ? nTimeBlocks() : b + 1;
    unsigned sStart = (s == -1) ? 0 : s;
    unsigned sEnd = (s == -1) ? this->nSubbands() : s + 1;
    unsigned pStart = (p == -1) ? 0 : p;
    unsigned pEnd = (p == -1) ? nPolarisations() : p + 1;

    QTextStream out(&file);
    for (unsigned b = bStart; b < bEnd; ++b) {
        for (unsigned s = sStart; s < sEnd; ++s) {
            for (unsigned p = pStart; p < pEnd; ++p) {

                // Get a pointer the the spectrum.
                data = spectrumData(b, s, p);

                for (unsigned c = 0; c < nChan; ++c)
                {
                    out << QString::number(data[c].real(), 'g', 8) << " ";
                    out << QString::number(data[c].imag(), 'g', 8) << endl;
                }

                out << endl;
            }
            out << endl;
        }
        out << endl;
    }
    file.close();
}


/**
 * @details
 * Returns the number of serialised bytes in the data blob when using
 * the serialise() method.
 */
quint64 SpectrumDataSetC32::serialisedBytes() const
{
    quint64 size = 4 * sizeof(unsigned);
    size += sizeof(double);
    size += sizeof(double);
    size += _data.size() * sizeof(std::complex<float>);
    return size;
}


/**
 * @details
 * Serialises the data blob.
 */
void SpectrumDataSetC32::serialise(QIODevice& out) const
{
    unsigned nBlocks = nTimeBlocks();
    unsigned nSubs = nSubbands();
    unsigned nPols = nPolarisations();
    unsigned nChan = nChannels();

    // Sub-band spectrum dimensions.
    out.write((char*)&nBlocks, sizeof(unsigned));
    out.write((char*)&nSubs, sizeof(unsigned));
    out.write((char*)&nPols, sizeof(unsigned));
    out.write((char*)&nChan, sizeof(unsigned));

    // Write the lofar meta-data.
    double blockRate = getBlockRate();
    double timeStamp = getLofarTimestamp();
    out.write((char*)&blockRate, sizeof(double));
    out.write((char*)&timeStamp, sizeof(double));

    // Write the data.
    out.write((char*)&_data[0], sizeof(std::complex<float>) * _data.size());
}


/**
 * @details
 * Deserialises the data blob.
 */
void SpectrumDataSetC32::deserialise(QIODevice& in, QSysInfo::Endian endian)
{
    if (endian != QSysInfo::ByteOrder) {
        throw QString("SubbandSpectraC32::deserialise(): Endianness "
                "of serial data not supported.");
    }

    unsigned nBlocks, nSubs, nPols, nChan;

    // Read spectrum dimensions.
    in.read((char*)&nBlocks, sizeof(unsigned));
    in.read((char*)&nSubs, sizeof(unsigned));
    in.read((char*)&nPols, sizeof(unsigned));
    in.read((char*)&nChan, sizeof(unsigned));

    // Read lofar meta-data
    double blockRate;
    double timeStamp;
    in.read((char*)&blockRate, sizeof(double));
    in.read((char*)&timeStamp, sizeof(double));
    setBlockRate(blockRate);
    setLofarTimestamp(timeStamp);

    // read the data
    resize(nBlocks, nSubs, nPols, nChan);
    in.read((char*)&_data[0], sizeof(std::complex<float>) * _data.size());
}






//------------------------------------------------------------------------------


/**
 * @details
 * Returns the number of serialised bytes in the data blob when using
 * the serialise() method.
 */
quint64 SpectrumDataSetStokes::serialisedBytes() const
{
    // Sub-band spactra dimensions.
    quint64 size = 4 * sizeof(unsigned);
    size += sizeof(double);
    size += sizeof(double);
    size += _data.size() * sizeof(float);
    return size;
}


/**
 * @details
 * Serialises the data blob.
 */
void SpectrumDataSetStokes::serialise(QIODevice& out) const
{
    unsigned nBlocks = nTimeBlocks();
    unsigned nSubs = nSubbands();
    unsigned nPols = nPolarisations();
    unsigned nChan = nChannels();

    // Sub-band spectrum dimensions.
    out.write((char*)&nBlocks, sizeof(unsigned));
    out.write((char*)&nSubs, sizeof(unsigned));
    out.write((char*)&nPols, sizeof(unsigned));
    out.write((char*)&nChan, sizeof(unsigned));

    // Write the lofar meta-data.
    double blockRate = getBlockRate();
    double timeStamp = getLofarTimestamp();
    out.write((char*)&blockRate, sizeof(double));
    out.write((char*)&timeStamp, sizeof(double));

    // Write the data.
    out.write((char*)&_data[0], sizeof(float) * _data.size());
}


/**
 * @details
 * Deserialises the data blob.
 */
void SpectrumDataSetStokes::deserialise(QIODevice& in, QSysInfo::Endian /*endian*/)
{
    unsigned nBlocks, nSubs, nPols, nChan;

    // Read spectrum dimensions.
    in.read((char*)&nBlocks, sizeof(unsigned));
    in.read((char*)&nSubs, sizeof(unsigned));
    in.read((char*)&nPols, sizeof(unsigned));
    in.read((char*)&nChan, sizeof(unsigned));

    // Read lofar meta-data
    double blockRate;
    double timeStamp;
    in.read((char*)&blockRate, sizeof(double));
    in.read((char*)&timeStamp, sizeof(double));
    setBlockRate(blockRate);
    setLofarTimestamp(timeStamp);

    // read the data
    resize(nBlocks, nSubs, nPols, nChan);
    in.read((char*)&_data[0], sizeof(float) * _data.size());
}


} // namespace lofar
} // namespace pelican

