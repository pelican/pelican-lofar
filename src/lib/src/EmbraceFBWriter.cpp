#include "SpectrumDataSet.h"
#include "EmbraceFBWriter.h"
#include "time.h"
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>

namespace pelican {
namespace ampp {


// Constructor
// TODO: For now we write in 32-bit format...
EmbraceFBWriter::EmbraceFBWriter(const ConfigNode& configNode )
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

    //    _nPols    = configNode.getOption("params", "nPolsToWrite", "1").toUInt();
    _nPols    = 1;
    _nchans   = _nChannels * _nSubbands;
    _buffSize = configNode.getOption("params", "bufferSize", "5120").toUInt();
    _cur = 0;
    _first = (configNode.hasAttribute("writeHeader") && configNode.getAttribute("writeHeader").toLower() == "true" );
    _site = configNode.getOption("TelescopeID", "value", "0").toUInt();
    _machine = configNode.getOption("MachineID", "value", "9").toUInt();

    _raStringX = configNode.getOption("RAJX", "value", "000000.0");
    _decStringX = configNode.getOption("DecJX", "value", "000000.0");
    _raX = _raStringX.toFloat();
    _decX = _decStringX.toFloat();
    _sourceNameX = _raStringX.left(4);
    if (_decX > 0.0 ) {
      _sourceNameX.append("+");
      _sourceNameX.append(_decStringX.left(4));
    } else {
      _sourceNameX.append(_decStringX.left(5));
    }

    _raStringY = configNode.getOption("RAJY", "value", "000000.0");
    _decStringY = configNode.getOption("DecJY", "value", "000000.0");
    _raY = _raStringY.toFloat();
    _decY = _decStringY.toFloat();
    _sourceNameY = _raStringY.left(4);
    if (_decY > 0.0 ) {
      _sourceNameY.append("+");
      _sourceNameY.append(_decStringY.left(4));
    } else {
      _sourceNameY.append(_decStringY.left(5));
    }
    // Open 2 files for writing X and Y direction

    _buffer.resize(_buffSize);

    char timestr[22];
    time_t     now = time(0);
    struct tm  tstruct;
    tstruct = *localtime(&now);
    strftime(timestr, sizeof timestr, "D%Y%m%dT%H%M%S", &tstruct );
    QString fileName;
    fileName = _filepath + QString("_") + timestr + QString("_X.dat");
    _file1.open(fileName.toUtf8().data(), std::ios::out | std::ios::binary);

    fileName = _filepath + QString("_") + timestr + QString("_Y.dat");
    _file2.open(fileName.toUtf8().data(), std::ios::out | std::ios::binary);
}

void EmbraceFBWriter::writeHeader(SpectrumDataSetStokes* stokes){
    double _timeStamp = stokes->getLofarTimestamp();
    struct tm tm;
    time_t _epoch;
    // MJD of 1/1/11 is 55562
    if ( strptime("2011-1-1 0:0:0", "%Y-%m-%d %H:%M:%S", &tm) != NULL ){
      _epoch = mktime(&tm);
    }
    else {
        throw( QString("EmbraceFBWriter: unable to set epoch.") );
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
    WriteString(_sourceNameX,_sourceNameY);
    WriteDouble("src_raj", _raX, _raY); // Write J2000 RA
    WriteDouble("src_dej", _decX, _decY); // Write J2000 Dec
    WriteDouble("fch1", _fch1);
    WriteDouble("foff", _foff);
    WriteInt("nchans", _nchans);
    WriteDouble("tsamp", _tsamp);
    WriteInt("nbits", _nBits);         // Only 32-bit binary data output is implemented for now
    WriteDouble("tstart", _mjdStamp);      //TODO: Extract start time from first packet
    WriteInt("nifs", int(_nPols));           // Polarisation channels.
    WriteString("HEADER_END");
    _file1.flush();
    _file2.flush();

}

// Destructor
EmbraceFBWriter::~EmbraceFBWriter()
{
    _file1.close();
    _file2.close();
}

// ---------------------------- Header helpers --------------------------
void EmbraceFBWriter::WriteString(QString string)
{
    int len = string.size();
    char *text = string.toUtf8().data();
    _file1.write(reinterpret_cast<char *>(&len), sizeof(int));
    _file1.write(reinterpret_cast<char *>(text), len);
    _file2.write(reinterpret_cast<char *>(&len), sizeof(int));
    _file2.write(reinterpret_cast<char *>(text), len);

}

void EmbraceFBWriter::WriteString(QString string1, QString string2)
{
    int len = string1.size();
    char *text = string1.toUtf8().data();
    _file1.write(reinterpret_cast<char *>(&len), sizeof(int));
    _file1.write(reinterpret_cast<char *>(text), len);
    len = string2.size();
    text = string2.toUtf8().data();
    _file2.write(reinterpret_cast<char *>(&len), sizeof(int));
    _file2.write(reinterpret_cast<char *>(text), len);

}

void EmbraceFBWriter::WriteInt(QString name, int value)
{
    WriteString(name);
    _file1.write(reinterpret_cast<char *>(&value), sizeof(int));
    _file2.write(reinterpret_cast<char *>(&value), sizeof(int));
}

void EmbraceFBWriter::WriteDouble(QString name, double value)
{
    WriteString(name);
    _file1.write(reinterpret_cast<char *>(&value), sizeof(double));
    _file2.write(reinterpret_cast<char *>(&value), sizeof(double));
}

void EmbraceFBWriter::WriteDouble(QString name, double value1, double value2)
{
    WriteString(name);
    _file1.write(reinterpret_cast<char *>(&value1), sizeof(double));
    _file2.write(reinterpret_cast<char *>(&value2), sizeof(double));
}

void EmbraceFBWriter::WriteLong(QString name, long value)
{
    WriteString(name);
    _file1.write(reinterpret_cast<char *>(&value), sizeof(long));
    _file2.write(reinterpret_cast<char *>(&value), sizeof(long));
}

// ---------------------------- Data helpers --------------------------

// Write data blob to disk
void EmbraceFBWriter::sendStream(const QString& /*streamName*/, const DataBlob* incoming)
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
	    for (int s = nSubbands - 1; s >= 0 ; --s) {
	      long index = stokes->index(s, nSubbands, 
					 0, nPolarisations, t, nChannels );
	      for(int i = nChannels - 1; i >= 0 ; --i) {
		_file1.write(reinterpret_cast<const char*>(&data[index + i]), 
			     sizeof(float));
	      }
	    }
	    for (int s = nSubbands - 1; s >= 0 ; --s) {
	      long index = stokes->index(s, nSubbands, 
					 1, nPolarisations, t, nChannels );
	      for(int i = nChannels - 1; i >= 0 ; --i) {
		_file2.write(reinterpret_cast<const char*>(&data[index + i]), 
			     sizeof(float));
	      }
	    }
	  }
	  break;
	}
	default:
	  throw(QString("EmbraceFBWriter: %1 bit datafiles not yet supported"));
	  break;
        }
	_file1.flush();
	_file2.flush();
    }
    else {
        std::cerr << "EmbraceFBWriter::send(): "
                "Only SpectrumDataSetStokes data can be written by the Writer" << std::endl;
        return;
    }
}

    // DO I NEED ANY OF THE REST??



void EmbraceFBWriter::_write(char* data, size_t size)
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

void EmbraceFBWriter::_float2int(const float *f, int *i)
{
    float ftmp;
    ftmp = (*f>_cropMax)? (_cropMax) : *f;
    *i = (ftmp<_cropMin) ? 0 : (int)rint((ftmp-_cropMin)*_nRange/_scaleDelta);
}

} // namepsace lofar
} // namepsace pelican
