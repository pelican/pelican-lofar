#include "SubbandSpectra.h"

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
void SubbandSpectraC32::write(const QString& fileName) const
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
                const std::complex<float>* spectrum = _data[index].ptr();
                unsigned nChannels = _data[index].nChannels();

                for (unsigned c = 0; c < nChannels; ++c) {
                    double re = spectrum[c].real();
                    double im = spectrum[c].imag();
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


/**
 * @details
 * Returns the number of serialised bytes in the data blob when using
 * the serialise() method.
 */
quint64 SubbandSpectraC32::serialisedBytes() const
{
    // Sub-band spactra dimensions.
    quint64 size = 3 * sizeof(unsigned);

    for (unsigned i = 0; i < _data.size(); ++i) {
        // Spectrum header.
        size += sizeof(unsigned) + 2 * sizeof(double);
        // Spectrum data.
        size += _data[i].nChannels() * sizeof(std::complex<float>);
    }
    return size;
}


/**
 * @details
 * Serialises the data blob.
 */
void SubbandSpectraC32::serialise(QIODevice& out) const
{
    // Sub-band spectrum dimensions.
    out.write((char*)&_nTimeBlocks, sizeof(unsigned));
    out.write((char*)&_nSubbands, sizeof(unsigned));
    out.write((char*)&_nPolarisations, sizeof(unsigned));

    // Loop over and write each spectrum.
    for (unsigned i = 0; i < _data.size(); ++i) {
        const std::complex<float>* spectrum = _data[i].ptr();
        unsigned nChannels = _data[i].nChannels();
        double startFreq = _data[i].startFrequency();
        double deltaFreq = _data[i].frequencyIncrement();
        // Spectrum header.
        out.write((char*)&nChannels, sizeof(unsigned));
        out.write((char*)&startFreq, sizeof(double));
        out.write((char*)&deltaFreq, sizeof(double));
        // Spectrum data.
        out.write((char*)&spectrum, nChannels * sizeof(std::complex<float>));
    }
}


/**
 * @details
 * Deserialises the data blob.
 */
void SubbandSpectraC32::deserialise(QIODevice& in, QSysInfo::Endian endian)
{
    if (endian != QSysInfo::ByteOrder) {
        throw QString("SubbandSpectraC32::deserialise(): Endianness "
                "of serial data not supported.");
    }

    // Read spectrum dimensions.
    in.read((char*)&_nTimeBlocks, sizeof(unsigned));
    in.read((char*)&_nSubbands, sizeof(unsigned));
    in.read((char*)&_nPolarisations, sizeof(unsigned));

    resize(_nTimeBlocks, _nSubbands, _nPolarisations);

    // Loop over and write each spectrum.
    for (unsigned i = 0; i < _data.size(); ++i) {

        // Read the spectrum header.
        unsigned nChannels = 0;
        double startFreq = 0.0;
        double deltaFreq = 0.0;
        in.read((char*)&nChannels, sizeof(unsigned));
        in.read((char*)&startFreq, sizeof(double));
        in.read((char*)&deltaFreq, sizeof(double));

        // Read the spectrum data.
        _data[i].resize(nChannels);
        in.read((char*)_data[i].ptr(),
                nChannels * sizeof(std::complex<float>));
    }
}

//------------------------------------------------------------------------------


/**
 * @details
 * Returns the number of serialised bytes in the data blob when using
 * the serialise() method.
 */
quint64 SubbandSpectraStokes::serialisedBytes() const
{
    // Sub-band spactra dimensions.
    quint64 size = 3 * sizeof(unsigned);
//    std::cout << "SubbandSpectraStokes::serialisedBytes(): nSpectra = " << _data.size() << std::endl;
//    std::cout << "SubbandSpectraStokes::serialisedBytes(): nTimeBlocks = " << nTimeBlocks() << std::endl;
//    std::cout << "SubbandSpectraStokes::serialisedBytes(): nSubbands = " << nSubbands() << std::endl;
//    std::cout << "SubbandSpectraStokes::serialisedBytes(): nPolarisations = " << nPolarisations() << std::endl;
//    std::cout << "SubbandSpectraStokes::serialisedBytes(): nChannels = " << _data[0].nChannels() << std::endl;
    for (unsigned i = 0; i < _data.size(); ++i) {
        // Spectrum header.
        size += sizeof(unsigned) + 2 * sizeof(double);
        // Spectrum data.
        size += _data[i].nChannels() * sizeof(float);
    }
//    std::cout << "SubbandSpectraStokes::serialisedBytes(): bytes = " << size << std::endl;
    return size;
}


/**
 * @details
 * Serialises the data blob.
 */
void SubbandSpectraStokes::serialise(QIODevice& out) const
{
    // Sub-band spectrum dimensions.
    out.write((char*)&_nTimeBlocks, sizeof(unsigned));
    out.write((char*)&_nSubbands, sizeof(unsigned));
    out.write((char*)&_nPolarisations, sizeof(unsigned));

    // Loop over and write each spectrum.
    for (unsigned i = 0; i < _data.size(); ++i) {
        unsigned nChannels = _data[i].nChannels();
        double startFreq = _data[i].startFrequency();
        double deltaFreq = _data[i].frequencyIncrement();
        // Spectrum header.
        out.write((char*)&nChannels, sizeof(unsigned));
        out.write((char*)&startFreq, sizeof(double));
        out.write((char*)&deltaFreq, sizeof(double));
        // Spectrum data.
        const float* spectrum = _data[i].ptr();
        out.write((char*)spectrum, nChannels * sizeof(float));
    }
}


/**
 * @details
 * Deserialises the data blob.
 */
void SubbandSpectraStokes::deserialise(QIODevice& in, QSysInfo::Endian /*endian*/)
{

    // TODO: the endianness parameter is broken somewhere...
//    if (endian != QSysInfo::ByteOrder) {
//        throw QString("SubbandSpectraStokes::deserialise(): Endianness "
//                "of serial data not supported.");
//    }

    // Read spectrum dimensions.
    in.read((char*)&_nTimeBlocks, sizeof(unsigned));
    in.read((char*)&_nSubbands, sizeof(unsigned));
    in.read((char*)&_nPolarisations, sizeof(unsigned));
    resize(_nTimeBlocks, _nSubbands, _nPolarisations);

    unsigned nChannels = 0;
    double startFreq = 0.0, deltaFreq = 0.0;

    // Loop over and write each spectrum.
    for (unsigned i = 0; i < _data.size(); ++i) {
        // Read the spectrum header.
        in.read((char*)&nChannels, sizeof(unsigned));
        in.read((char*)&startFreq, sizeof(double));
        in.read((char*)&deltaFreq, sizeof(double));
        // Read the spectrum data.
        _data[i].resize(nChannels);
        in.read((char*)_data[i].ptr(), nChannels * sizeof(float));
    }
}


} // namespace lofar
} // namespace pelican

