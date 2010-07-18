#include "SubbandSpectra.h"
#include "SigprocStokesWriter.h"

#include <iostream>
#include <fstream>

namespace pelican {
namespace lofar {


// Constructor
// TODO: For now we write in 32-bit format...
SigprocStokesWriter::SigprocStokesWriter(const ConfigNode& configNode )
: AbstractOutputStream(configNode)
{
    // Initliase connection manager thread
    _filepath = configNode.getOption("file", "filepath");
    _fch1     = configNode.getOption("params", "channelOneFrequency", "0").toFloat();
    _foff     = configNode.getOption("params", "frequencyOffset", "0").toFloat();
    _tsamp    = configNode.getOption("params", "samplingTime", "0").toFloat();
    _nchans   = configNode.getOption("params", "numberOfChannels", "31").toFloat();
    _nPols    = configNode.getOption("params", "nPols", "1").toUInt();

    // Open file
    _file.open(_filepath.toUtf8().data(), std::ios::out | std::ios::binary);

    // Write header
    WriteString("HEADER_START");
    //WriteString("Telescope");
    //WriteString("LOFAR");
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
void SigprocStokesWriter::send(const QString& /*streamName*/, const DataBlob* incoming)
{
    SubbandSpectraStokes* stokes;
    DataBlob* blob = const_cast<DataBlob*>(incoming);

    if (dynamic_cast<SubbandSpectraStokes*>(blob)) {
        stokes = (SubbandSpectraStokes*) dynamic_cast<SubbandSpectraStokes*>(blob);

        unsigned nSamples = stokes->nTimeBlocks();
        unsigned nSubbands = stokes->nSubbands();
//        unsigned nPolarisations = stokes->nPolarisations(); // this is now an option.
        unsigned nChannels = stokes->ptr(0, 0, 0)->nChannels();
        float* data;
        size_t dataSize = nChannels * sizeof(float);

        for (unsigned t = 0; t < nSamples; ++t) {
            for (unsigned p = 0; p < _nPols; ++p) {
                for (int s = nSubbands - 1; s >= 0; --s) {
                    data = stokes->ptr(t, s, p)->ptr();
	            for(int i = dataSize - 1; i >= 0 ; --i )
		    {
	               //_file.write(reinterpret_cast<char*>(data), dataSize);
	               _file.write(reinterpret_cast<char* >(&data[i]), sizeof(float));
	            }
                    _file.flush();
                }
            }
        }
    }
    else {
        std::cerr << "SigprocStokesWriter::send(): "
                "Only  blobs can be written by the SigprocWriter" << std::endl;
        return;
    }
}

}
}
