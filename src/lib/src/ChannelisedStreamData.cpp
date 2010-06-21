#include "ChannelisedStreamData.h"
#include <QtCore/QIODevice>
#include <QtCore/QTextStream>
#include <QtCore/QFile>
#include <QtCore/QString>

#include <iostream>

namespace pelican {
namespace lofar {

/**
 * @details
 * Writes the contents of the channelised data blob to a ASCII file.
 *
 * @param fileName The name of the file to write to.
 */
void ChannelisedStreamData::write(const QString& fileName) const
{
    QFile file(fileName);
	if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
		return;
	}
	QTextStream out(&file);
	for (unsigned index = 0, s = 0; s < _nSubbands; ++s) {
		for (unsigned p = 0; p < _nPolarisations; ++p) {
			for (unsigned c = 0; c < _nChannels; ++c) {
				double re = _data[index].real();
				double im = _data[index].imag();
				out << QString::number(re, 'g', 16) << " ";
				out << QString::number(im, 'g', 16) << " ";
				index++;
			}
			out << endl;
		}
		out << endl;
	}
	file.close();
}


/**
 * @details
 * Serialises the data blob.
 */
void ChannelisedStreamData::serialise(QIODevice& out) const
{
    // Header.
    // TODO: Add some sort of time stamp? (from the data blob base class.)
    out.write((char*)&_nSubbands, sizeof(unsigned));
    out.write((char*)&_nPolarisations, sizeof(unsigned));
    out.write((char*)&_nChannels, sizeof(unsigned));
    out.write((char*)&_startFreq, sizeof(double));
    out.write((char*)&_channelFreqDelta, sizeof(double));

    // Data.
    out.write((char*)&_data[0], _data.size() * sizeof(std::complex<double>));
}


/**
 * @details
 * Deserialises the data blob.
 */
void ChannelisedStreamData::deserialise(QIODevice& in)
{
    // Read the header.
    unsigned offset = 0;
    in.read((char*)&_nSubbands, sizeof(unsigned));
    in.read((char*)&_nPolarisations, sizeof(unsigned));
    in.read((char*)&_nChannels, sizeof(unsigned));
    in.read((char*)&_startFreq, sizeof(double));
    in.read((char*)&_channelFreqDelta, sizeof(double));

    unsigned dataPoints = _nSubbands * _nPolarisations * _nChannels;
    unsigned dataSize = dataPoints * sizeof(std::complex<double>);

    // Resize the data blob to fit the byte array.
    resize(_nSubbands, _nPolarisations, _nChannels);

    // Read the data.
    in.read((char*)&_data[0], dataSize);
}

	} // namespace lofar
} // namespace pelican

