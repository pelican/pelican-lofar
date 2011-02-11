#include "SpectrumDataSet.h"
#include "SigprocStokesWriter.h"

#include <string>
#include <cstring>
#include <iostream>
#include <fstream>

namespace pelican {
namespace lofar {


// Constructor
// TODO: For now we write in 32-bit format...
SigprocStokesWriter::SigprocStokesWriter(const ConfigNode& configNode )
: AbstractOutputStream(configNode)
{
    _nSubbands = configNode.getOption("subbandsPerPacket", "value", "1").toUInt();
    _nTotalSubbands = configNode.getOption("totalComplexSubbands", "value", "1").toUInt();
    _clock = configNode.getOption("clock", "value", "200").toFloat();
    _integration    = configNode.getOption("integrateTimeBins", "value", "1").toUInt();
    _nChannels = configNode.getOption("outputChannelsPerSubband", "value", "128").toUInt();
    _nRawPols = configNode.getOption("nRawPolarisations", "value", "2").toUInt();

    // Initliase connection manager thread
    _filepath = configNode.getOption("file", "filepath");
    _topsubband     = configNode.getOption("topSubbandIndex", "value", "150").toFloat();
    _lbahba     = configNode.getOption("LBA_0_or_HBA_1", "value", "1").toFloat();
    _fch1     = _lbahba * 100 + _clock / (_nRawPols * _nTotalSubbands) * _topsubband;
    _foff     = -_clock / (_nRawPols * _nTotalSubbands) / float(_nChannels);
    _tsamp    = (_nRawPols * _nTotalSubbands) * _nChannels * _integration / _clock/ 1e6;
    _nPols    = configNode.getOption("params", "nPolsToWrite", "1").toUInt();
    _nchans   = _nChannels * _nSubbands;
    _buffSize = configNode.getOption("params", "bufferSize", "5120").toUInt();
    _cur = 0;

    // Open file
    _buffer.resize(_buffSize);
    _file.open(_filepath.toUtf8().data(), std::ios::out | std::ios::binary);

    // Write header
    if( configNode.hasAttribute("writeHeader") && configNode.getAttribute("writeHeader").toLower() == "true" ) {
        WriteString("HEADER_START");
        WriteInt("machine_id", 0);    // Ignore for now
        WriteInt("telescope_id", 0);  // Ignore for now
        WriteInt("data_type", 1);     // Channelised Data

        // Need to be parametrised ...
        WriteDouble("fch1", _fch1);
        WriteDouble("foff", _foff);
        WriteInt("nchans", _nchans);
        WriteDouble("tsamp", _tsamp);
        WriteInt("nbits", 32);         // Only 32-bit binary data output is implemented for now
        WriteDouble("tstart", 0);      //TODO: Extract start time from first packet
        WriteInt("nifs", int(_nPols));		   // Polarisation channels.
        WriteString("HEADER_END");
        _file.flush();
    }
}

// Destructor
SigprocStokesWriter::~SigprocStokesWriter()
{
    _file.close();
}

// ---------------------------- Header helpers --------------------------
void SigprocStokesWriter::WriteString(QString string)
{
    int len = string.size();
    char *text = string.toUtf8().data();
    _file.write(reinterpret_cast<char *>(&len), sizeof(int));
    _file.write(reinterpret_cast<char *>(text), len);
}

void SigprocStokesWriter::WriteInt(QString name, int value)
{
    WriteString(name);
    _file.write(reinterpret_cast<char *>(&value), sizeof(int));
}

void SigprocStokesWriter::WriteDouble(QString name, double value)
{
    WriteString(name);
    _file.write(reinterpret_cast<char *>(&value), sizeof(double));
}

void SigprocStokesWriter::WriteLong(QString name, long value)
{
    WriteString(name);
    _file.write(reinterpret_cast<char *>(&value), sizeof(long));
}

// ---------------------------- Data helpers --------------------------

// Write data blob to disk
void SigprocStokesWriter::sendStream(const QString& /*streamName*/, const DataBlob* incoming)
{
    SpectrumDataSetStokes* stokes;
    DataBlob* blob = const_cast<DataBlob*>(incoming);

    if (dynamic_cast<SpectrumDataSetStokes*>(blob)) {
        stokes = (SpectrumDataSetStokes*) dynamic_cast<SpectrumDataSetStokes*>(blob);

        unsigned nSamples = stokes->nTimeBlocks();
        unsigned nSubbands = stokes->nSubbands();
        unsigned nChannels = stokes->nChannels();
        float const * data;

        for (unsigned t = 0; t < nSamples; ++t) {
            for (unsigned p = 0; p < _nPols; ++p) {
                for (int s = nSubbands - 1; s >= 0 ; --s) {
                    data = stokes->spectrumData(t, s, p);
                    for(int i = nChannels - 1; i >= 0 ; --i)
                        _file.write(reinterpret_cast<const char*>(&data[i]), sizeof(float));
                }
            }
        }

        _file.flush();
    }
    else {
        std::cerr << "SigprocStokesWriter::send(): "
                "Only blobs can be written by the SigprocWriter" << std::endl;
        return;
    }
}

void SigprocStokesWriter::_write(char* data, size_t size)
{
    int max = _buffer.capacity() -1;
    int ptr = (_cur + size) % max;
    if( ptr <= _cur ) {
        int dsize = max - _cur;
        std::memcpy(&_buffer[_cur], data, dsize );
        _file.write(&_buffer[0], max);
        _cur = 0; size -= dsize; data += dsize * sizeof(char);
    }
    std::cout << "max=" << max << " ptr=" << ptr << " cur=" << _cur << " size=" << size << "data=" << data << std::endl;
    std::memcpy( &_buffer[_cur], data, size);
    _cur=ptr;
}

}
}
