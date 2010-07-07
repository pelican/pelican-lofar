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
                const std::complex<float>* spectrum = _spectra[index].ptr();
                unsigned nChannels = _spectra[index].nChannels();

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

    for (unsigned i = 0; i < _spectra.size(); ++i) {
        // Spectrum header.
        size += sizeof(unsigned) + 2 * sizeof(double);
        // Spectrum data.
        size += _spectra[i].nChannels() * sizeof(std::complex<float>);
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
    for (unsigned i = 0; i < _spectra.size(); ++i) {
        const std::complex<float>* spectrum = _spectra[i].ptr();
        unsigned nChannels = _spectra[i].nChannels();
        double startFreq = _spectra[i].startFrequency();
        double deltaFreq = _spectra[i].channelFrequencyDelta();
        // Spectrum header.
        out.write((char*)&nChannels, sizeof(unsigned));
        out.write((char*)&startFreq, sizeof(double));
        out.write((char*)&deltaFreq, sizeof(double));
        // Spectrum data.
        out.write((char*)&spectrum,
                nChannels * sizeof(std::complex<float>));
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
    for (unsigned i = 0; i < _spectra.size(); ++i) {

        // Read the spectrum header.
        unsigned nChannels = 0;
        double startFreq = 0.0;
        double deltaFreq = 0.0;
        in.read((char*)&nChannels, sizeof(unsigned));
        in.read((char*)&startFreq, sizeof(double));
        in.read((char*)&deltaFreq, sizeof(double));

        // Read the spectrum data.
        _spectra[i].resize(nChannels);
        in.read((char*)_spectra[i].ptr(),
                nChannels * sizeof(std::complex<float>));
    }
}


} // namespace lofar
} // namespace pelican

