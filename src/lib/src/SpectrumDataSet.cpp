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
void SpectrumDataSetC32::write(const QString& fileName) const
{
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return;

    const std::complex<float>* data;
    unsigned nChan = nChannels();

    QTextStream out(&file);
    for (unsigned b = 0; b < nTimeBlocks(); ++b) {
        for (unsigned s = 0; s < nSubbands(); ++s) {
            for (unsigned p = 0; p < nPolarisations(); ++p) {

                // Get a pointer the the spectrum.
                data = spectrumData(b, s, p);

                for (unsigned c = 0; c < nChan; ++c)
                {
                    out << QString::number(data[c].real(), 'g', 16) << " ";
                    out << QString::number(data[c].imag(), 'g', 16) << endl;
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
    // Sub-band spectra dimensions.
    quint64 size = 3 * sizeof(unsigned);

    unsigned nChan = nChannels();
    for (unsigned i = 0; i < nSpectra(); ++i) {
        // Spectrum header.
        size += sizeof(unsigned) + 2 * sizeof(double);
        // Spectrum data.
        size += nChan * sizeof(std::complex<float>);
    }
    return size;
}


/**
 * @details
 * Serialises the data blob.
 */
void SpectrumDataSetC32::serialise(QIODevice& out) const
{
//    unsigned nBlocks = nTimeBlocks();
//    unsigned nSubs = nSubbands();
//    unsigned nPols = nPolarisations();
//
//    // Sub-band spectrum dimensions.
//    out.write((char*)&nBlocks, sizeof(unsigned));
//    out.write((char*)&nSubs, sizeof(unsigned));
//    out.write((char*)&nPols, sizeof(unsigned));
//
//    Spectrum<std::complex<float> > const * spectrum;
//
//    double startFreq, deltaFreq;
//    unsigned nChan = nChannels();
//
//    // Loop over and write each spectrum.
//    for (unsigned i = 0; i < nSpectra(); ++i) {
//        spectrum = this->spectrum(i);
//        startFreq = spectrum->startFrequency();
//        deltaFreq = spectrum->frequencyIncrement();
//        // Spectrum header.
//        out.write((char*)&nChan, sizeof(unsigned));
//        out.write((char*)&startFreq, sizeof(double));
//        out.write((char*)&deltaFreq, sizeof(double));
//        // Spectrum data.
//        out.write((char*)spectrum->data(), nChan * sizeof(std::complex<float>));
//    }
}


/**
 * @details
 * Deserialises the data blob.
 */
void SpectrumDataSetC32::deserialise(QIODevice& in, QSysInfo::Endian endian)
{
//    if (endian != QSysInfo::ByteOrder) {
//        throw QString("SubbandSpectraC32::deserialise(): Endianness "
//                "of serial data not supported.");
//    }
//
//    unsigned nBlocks, nSubs, nPols;
//
//    // Read spectrum dimensions.
//    in.read((char*)&nBlocks, sizeof(unsigned));
//    in.read((char*)&nSubs, sizeof(unsigned));
//    in.read((char*)&nPols, sizeof(unsigned));
//
//    resize(nBlocks, nSubs, nPols);
//
//    unsigned nChannels;
//    double startFreq, deltaFreq;
//    Spectrum<std::complex<float> >* spectrum;
//
//    // Loop over and write each spectrum.
//    for (unsigned i = 0; i < nSpectra(); ++i) {
//
//        spectrum = this->spectrum(i);
//
//        // Read the spectrum header.
//        in.read((char*)&nChannels, sizeof(unsigned));
//        in.read((char*)&startFreq, sizeof(double));
//        in.read((char*)&deltaFreq, sizeof(double));
//        spectrum->setStartFrequency(startFreq);
//        spectrum->setFrequencyIncrement(deltaFreq);
//
//        // Read the spectrum data.
//        spectrum->resize(nChannels);
//        in.read((char*)spectrum->data(), nChannels * sizeof(std::complex<float>));
//    }
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
    quint64 size = 3 * sizeof(unsigned);

    unsigned nChan = nChannels();
    for (unsigned i = 0; i < nSpectra(); ++i) {
        // Spectrum header.
        size += sizeof(unsigned) + 2 * sizeof(double);
        // Spectrum data.
        size += nChan * sizeof(float);
    }
    return size;
}


/**
 * @details
 * Serialises the data blob.
 */
void SpectrumDataSetStokes::serialise(QIODevice& out) const
{
//    unsigned nBlocks = nTimeBlocks();
//    unsigned nSubs = nSubbands();
//    unsigned nPols = nPolarisations();
//
//    // Sub-band spectrum dimensions.
//    out.write((char*)&nBlocks, sizeof(unsigned));
//    out.write((char*)&nSubs, sizeof(unsigned));
//    out.write((char*)&nPols, sizeof(unsigned));
//
//    Spectrum<float> const * spectrum;
//    double startFreq, deltaFreq;
//    unsigned nChan = nChannels();
//
//    // Loop over and write each spectrum.
//    for (unsigned i = 0; i < nSpectra(); ++i) {
//        spectrum = this->spectrum(i);
//        startFreq = spectrum->startFrequency();
//        deltaFreq = spectrum->frequencyIncrement();
//        // Spectrum header.
//        out.write((char*)&nChan, sizeof(unsigned));
//        out.write((char*)&startFreq, sizeof(double));
//        out.write((char*)&deltaFreq, sizeof(double));
//        // Spectrum data.
//        out.write((char*)spectrum->data(), nChan * sizeof(float));
//    }
}


/**
 * @details
 * Deserialises the data blob.
 */
void SpectrumDataSetStokes::deserialise(QIODevice& in, QSysInfo::Endian /*endian*/)
{
//    unsigned nBlocks, nSubs, nPols;
//
//    // Read spectrum dimensions.
//    in.read((char*)&nBlocks, sizeof(unsigned));
//    in.read((char*)&nSubs, sizeof(unsigned));
//    in.read((char*)&nPols, sizeof(unsigned));
//
//    resize(nBlocks, nSubs, nPols, nChannels);
//
//    unsigned nChannels;
//    double startFreq, deltaFreq;
//    Spectrum<float>* spectrum;
//
//    // Loop over and write each spectrum.
//    for (unsigned i = 0; i < nSpectra(); ++i) {
//
//        spectrum = this->spectrum(i);
//
//        // Read the spectrum header.
//        in.read((char*)&nChannels, sizeof(unsigned));
//        in.read((char*)&startFreq, sizeof(double));
//        in.read((char*)&deltaFreq, sizeof(double));
//        spectrum->setStartFrequency(startFreq);
//        spectrum->setFrequencyIncrement(deltaFreq);
//
//        // Read the spectrum data.
//        spectrum->resize(nChannels);
//        in.read((char*)spectrum->data(), nChannels * sizeof(float));
//    }
}


} // namespace lofar
} // namespace pelican

