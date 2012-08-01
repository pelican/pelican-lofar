#include "SpectrumDataSet.h"
#include "SigprocStokesWriter.h"
#include "time.h"
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>

namespace pelican {
namespace lofar {


// Constructor
// TODO: For now we write in 32-bit format...
SigprocStokesWriter::SigprocStokesWriter(const ConfigNode& configNode )
  : AbstractOutputStream(configNode), _first(true)
{
    _nSubbands = configNode.getOption("subbandsPerPacket", "value", "1").toUInt();
    _nTotalSubbands = configNode.getOption("totalComplexSubbands", "value", "1").toUInt();
    _clock = configNode.getOption("clock", "value", "200").toFloat();
    _integration    = configNode.getOption("integrateTimeBins", "value", "1").toUInt();
    _nChannels = configNode.getOption("outputChannelsPerSubband", "value", "128").toUInt();
    _nRawPols = configNode.getOption("nRawPolarisations", "value", "2").toUInt();
    _nBits = configNode.getOption("dataBits", "value", "32").toUInt();
    _nRange = (int) pow(2.0,(double) _nBits)-1.0;
    _cropMin = configNode.getOption("scale", "min", "0.0" ).toFloat();
    bool goodConversion=false;
    _cropMax = configNode.getOption("scale", "max", "X" ).toFloat(&goodConversion);
    if( ! goodConversion ) {
        _cropMax=_nRange;
    }
    _scaleDelta = _cropMax - _cropMin;

    // Initliase connection manager thread
    _filepath = configNode.getOption("file", "filepath");
    _topsubband     = configNode.getOption("topSubbandIndex", "value", "150").toFloat();
    _lbahba     = configNode.getOption("LBA_0_or_HBA_1", "value", "1").toFloat();
    if( configNode.getOption("fch1", "value" ) == "" ){ 
      if (_lbahba == 0) {
        _fch1     = _clock / (_nRawPols * _nTotalSubbands) * _topsubband;
      }
      else{
        if (_clock == 200)
          _fch1     = 100 + _clock / (_nRawPols * _nTotalSubbands) * _topsubband;
        if (_clock == 160)
          _fch1     = 160 + _clock / (_nRawPols * _nTotalSubbands) * _topsubband;
      }
    }
    else{
      _fch1     = configNode.getOption("fch1", "value", "1400.0").toFloat();
    }
    if( configNode.getOption("foff", "value" ) == "" ){ 
      _foff     = -_clock / (_nRawPols * _nTotalSubbands) / float(_nChannels);
    }
    else{
      _foff     = configNode.getOption("foff", "value", "1.0").toFloat();
    }
    if( configNode.getOption("tsamp", "value" ) == "" ){ 
      _tsamp    = (_nRawPols * _nTotalSubbands) * _nChannels * _integration / _clock/ 1e6;
    }
    else{
      _tsamp     = configNode.getOption("tsamp", "value", "1.0").toFloat();
    }

    _nPols    = configNode.getOption("params", "nPolsToWrite", "1").toUInt();
    _nchans   = _nChannels * _nSubbands;
    _buffSize = configNode.getOption("params", "bufferSize", "5120").toUInt();
    _cur = 0;
    _first = (configNode.hasAttribute("writeHeader") && configNode.getAttribute("writeHeader").toLower() == "true" );
    _site = configNode.getOption("TelescopeID", "value", "0").toUInt();
    _machine = configNode.getOption("MachineID", "value", "9").toUInt();
    _raString = configNode.getOption("RAJ", "value", "000000.0");
    _decString = configNode.getOption("DecJ", "value", "000000.0");
    _ra = _raString.toFloat();
    _dec = _decString.toFloat();
    _sourceName = _raString.left(4);
    if (_dec > 0.0 ) {
      _sourceName.append("+");
      _sourceName.append(_decString.left(4));
    } else {
      _sourceName.append(_decString.left(5));
    }
    // Open file
    _buffer.resize(_buffSize);

    char timestr[22];
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(timestr, sizeof timestr, "D%Y%m%dT%H%M%S", &tstruct );
    QString fileName;
    fileName = _filepath + QString("_") + timestr + QString(".dat");
    //    _file.open(_filepath.toUtf8().data(), std::ios::out | std::ios::binary);
    _file.open(fileName.toUtf8().data(), std::ios::out | std::ios::binary);
}

void SigprocStokesWriter::writeHeader(SpectrumDataSetStokes* stokes){
    double _timeStamp = stokes->getLofarTimestamp();
    struct tm tm;
    time_t _epoch;
    // MJD of 1/1/11 is 55562
    if ( strptime("2011-1-1 1:0:0", "%Y-%m-%d %H:%M:%S", &tm) != NULL ){
      _epoch = mktime(&tm);
    }
    else {
        throw( QString("SigprocStokesWriter: unable to set epoch.") );
    }
    double _mjdStamp = (_timeStamp-_epoch)/86400 + 55562.0;
    std::cout << "MJD timestamp:" << std::fixed << _mjdStamp << std::endl;

    // Write header
    WriteString("HEADER_START");
    WriteInt("machine_id", _machine);    // Ignore for now
    WriteInt("telescope_id", _site);
    WriteInt("data_type", 1);     // Channelised Data

    // Need to be parametrised ...
    WriteString("source_name");
    WriteString(_sourceName);
    WriteDouble("src_raj", _ra); // Write J2000 RA
    WriteDouble("src_dej", _dec); // Write J2000 Dec
    WriteDouble("fch1", _fch1);
    WriteDouble("foff", _foff);
    WriteInt("nchans", _nchans);
    WriteDouble("tsamp", _tsamp);
    WriteInt("nbits", _nBits);         // Only 32-bit binary data output is implemented for now
    WriteDouble("tstart", _mjdStamp);      //TODO: Extract start time from first packet
    WriteInt("nifs", int(_nPols));           // Polarisation channels.
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
void SigprocStokesWriter::sendStream(const QString& /*streamName*/, const DataBlob* incoming)
{
    SpectrumDataSetStokes* stokes;
    DataBlob* blob = const_cast<DataBlob*>(incoming);

    if( (stokes = (SpectrumDataSetStokes*) dynamic_cast<SpectrumDataSetStokes*>(blob))){

        if (_first){
            _first = false;
            writeHeader(stokes);
        }

        unsigned nSamples = stokes->nTimeBlocks();
        unsigned nSubbands = stokes->nSubbands();
        unsigned nChannels = stokes->nChannels();
        unsigned nPolarisations = stokes->nPolarisations();
        float const * data = stokes->data();

        switch (_nBits) {
            case 32: {
                    for (unsigned t = 0; t < nSamples; ++t) {
                        for (unsigned p = 0; p < _nPols; ++p) {
                            for (int s = nSubbands - 1; s >= 0 ; --s) {
                                long index = stokes->index(s, nSubbands, 
                                          p, nPolarisations, t, nChannels );
                                for(int i = nChannels - 1; i >= 0 ; --i) {
                                _file.write(reinterpret_cast<const char*>(&data[index + i]), 
                                            sizeof(float));
                                }
                            }
                        }
                    }
                }
                break;
            case 8: {
                    for (unsigned t = 0; t < nSamples; ++t) {
                        for (unsigned p = 0; p < _nPols; ++p) {
                            for (int s = nSubbands - 1; s >= 0 ; --s) {
                                long index = stokes->index(s, nSubbands, 
                                          p, nPolarisations, t, nChannels );
                                for(int i = nChannels - 1; i >= 0 ; --i) {
                                    int ci;
                                    _float2int(&data[index + i],&ci);
                                    _file.write((const char*)&ci,sizeof(unsigned char));
                                }
                            }
                        }
                    }
                }
                break;
            default:
                throw(QString("SigprocStokesWriter: %1 bit datafiles not yet supported"));
                break;
        }
/*
        for (unsigned t = 0; t < nSamples; ++t) {
            for (unsigned p = 0; p < _nPols; ++p) {
                for (int s = nSubbands - 1; s >= 0 ; --s) {
                    data = stokes->spectrumData(t, s, p);
                    for(int i = nChannels - 1; i >= 0 ; --i) {
                        switch (_nBits) {
                            case 32:
                                _file.write(reinterpret_cast<const char*>(&data[i]), sizeof(float));
                                break;
                            case 8:
                                int ci = ;
                                _float2int(&data[i],1,8,_scaleMin,_scaleMax,&ci);
                                _file.write((const char*)&ci,sizeof(unsigned char));
                                break;
                            default:
                                throw(QString("SigprocStokesWriter:"));
                                break;
                        }
                    }
                }
            }
        }
*/
        _file.flush();
    }
    else {
        std::cerr << "SigprocStokesWriter::send(): "
                "Only SpectrumDataSetStokes data can be written by the SigprocWriter" << std::endl;
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

void SigprocStokesWriter::_float2int(const float *f, int *i)
{
    float ftmp;
    ftmp = (*f>_cropMax)? (_cropMax) : *f;
    *i = (ftmp<_cropMin) ? 0 : (int)rint((ftmp-_cropMin)*_nRange/_scaleDelta);
}

} // namepsace lofar
} // namepsace pelican
