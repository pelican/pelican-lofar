#include "SpectrumDataSet.h"
#include "H5Writer.h"
#include "dal/dal_config.h"

#include <string>
#include <cstring>
#include <iostream>
#include <fstream>

namespace pelican {
namespace lofar {


// Constructor
// TODO: For now we write in 32-bit format...
H5Writer::H5Writer(const ConfigNode& configNode )
  : AbstractOutputStream(configNode), _first(true)
{
    _filepath = configNode.getOption("file", "filepath");
    _datatype = configNode.getOption("Stokes_0_or_Voltages_1", "value", "1").toUInt();

    // By definition for LOFAR RSP boards, the following should not change:
    _nRawPols = configNode.getOption("nRawPolarisations", "value", "2").toUInt();
    _nTotalSubbands = configNode.getOption("totalComplexSubbands", "value", "512").toUInt();


    // Parameters that change for every observation
    _nChannels = configNode.getOption("outputChannelsPerSubband", "value", "128").toUInt();
    _nSubbands = configNode.getOption("subbandsPerPacket", "value", "1").toUInt();
    _topsubband     = configNode.getOption("topSubbandIndex", "value", "150").toFloat();
    _integration    = configNode.getOption("integrateTimeBins", "value", "1").toUInt();
    _nBits = configNode.getOption("dataBits", "value", "32").toUInt();

    // For LOFAR, either 160 or 200, usually 200
    _clock = configNode.getOption("clock", "value", "200").toFloat();

    // LBA or HBA switch
    _lbahba     = configNode.getOption("LBA_0_or_HBA_1", "value", "1").toFloat();

    // Work out the top observing frequency (fch1) and channel bandwidth (foff)
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

    // Observation sampling time
    if( configNode.getOption("tsamp", "value" ) == "" ){ 
      _tsamp    = (_nRawPols * _nTotalSubbands) * _nChannels * _integration / _clock/ 1e6;
    }
    else{
      _tsamp     = configNode.getOption("tsamp", "value", "1.0").toFloat();
    }

    // Number of polarisations to write out, 1 for total power or 4
    // for stokes and complex voltages
    _nPols    = configNode.getOption("params", "nPolsToWrite", "1").toUInt();

    // Number of total numbers
    _nchans   = _nChannels * _nSubbands;

    _buffSize = configNode.getOption("params", "bufferSize", "5120").toUInt();

    _cur = 0;

    _first = (configNode.hasAttribute("writeHeader") && configNode.getAttribute("writeHeader").toLower() == "true" );

    // Observation specific
    // Chilbolton:14 Nancay:12 Effelsberg:13
    _site = configNode.getOption("TelescopeID", "value", "14").toUInt();
    _telescope = configNode.getOption("TelescopeName", "value", "UK608");
    _machine = configNode.getOption("MachineID", "value", "10").toUInt();
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

    _buffer.resize(_buffSize);

}

void H5Writer::writeHeader(SpectrumDataSetStokes* stokes){
    double _timeStamp = stokes->getLofarTimestamp();
    struct tm tm;
    time_t _epoch;
    // MJD of 1/1/11 is 55562
    if ( strptime("2011-1-1 1:0:0", "%Y-%m-%d %H:%M:%S", &tm) != NULL ){
      _epoch = mktime(&tm);
    }
    else {
        throw( QString("H5Writer: unable to set epoch.") );
    }
    double _mjdStamp = (_timeStamp-_epoch)/86400 + 55562.0;
    std::cout << "MJD timestamp:" << std::fixed << _mjdStamp << std::endl;


    // Create a total of _nPols h5 files for writing
    
    for (unsigned i=0; i<_nPols; ++i){
      QString singlePolFile;
      singlePolFile = _filepath;
      singlePolFile.append("_S0");
      singlePolFile.append(i);
      singlePolFile.append(".h5");

      //-------------- File  -----------------
/*
      DAL::BF_File* bf_file = new DAL::BF_File (singlePolFile.toUtf8().data(), "CREATE");
      std::cout << "Opened: " << singlePolFile << " for writing" << std::endl;
      Attribute<std::string> telescope = bf_file->telescope();
      telescope.set(_telescope.toUtf8().data());
      Attribute<unsigned> nsap = bf_file->nofSubArrayPointings();
      nsap.set(1); // Single subarray per single station for now
      
      //-------------- subarray pointing  -----------------
      BF_SubArrayPointing sap = bf_file->subArrayPointing (0);
      sap.create();
      Attribute<unsigned> nbeam = sap.nofBeams();
      nbeam.set(1); // Single beam per file for now
      
      //-------------- Beam -----------------
      BF_BeamGroup beam = sap.beam (0);
      beam.create();
      Attribute< std::vector<std::string> > targets = beam.target();
      std::vector<std::string> mytargets;
      mytargets.append(_sourceName.toUtf8().data());  // 1 target only
      targets.set(mytargets);
      Attribute<bool> volts = beam.complexVoltages();
      volts.set(_datatype); // Bool vs Uint

      // Coordinates within Beam
      CoordinatesGroup coord = beam.coordinates();
      coord.create();
      Attribute< std::vector<std::string> > types = coord.coordinateTypes();
      std::vector<std::string> t;
      t.append="Time";
      t.append="Spectral";
      types.set(t);
*/

    }
    // Fill up the relevant DAL variables



















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
H5Writer::~H5Writer()
{
    _file.close();
}

// ---------------------------- Header helpers --------------------------
void H5Writer::WriteString(QString string)
{
    int len = string.size();
    char *text = string.toUtf8().data();
    _file.write(reinterpret_cast<char *>(&len), sizeof(int));
    _file.write(reinterpret_cast<char *>(text), len);
}

void H5Writer::WriteInt(QString name, int value)
{
    WriteString(name);
    _file.write(reinterpret_cast<char *>(&value), sizeof(int));
}

void H5Writer::WriteDouble(QString name, double value)
{
    WriteString(name);
    _file.write(reinterpret_cast<char *>(&value), sizeof(double));
}

void H5Writer::WriteLong(QString name, long value)
{
    WriteString(name);
    _file.write(reinterpret_cast<char *>(&value), sizeof(long));
}

// ---------------------------- Data helpers --------------------------

// Write data blob to disk
void H5Writer::sendStream(const QString& /*streamName*/, const DataBlob* incoming)
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
                throw(QString("H5Writer: %1 bit datafiles not yet supported"));
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
                                throw(QString("H5Writer:"));
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
        std::cerr << "H5Writer::send(): "
                "Only SpectrumDataSetStokes data can be written by the SigprocWriter" << std::endl;
        return;
    }
}

void H5Writer::_write(char* data, size_t size)
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

void H5Writer::_float2int(const float *f, int *i)
{
    float ftmp;
    ftmp = (*f>_cropMax)? (_cropMax) : *f;
    *i = (ftmp<_cropMin) ? 0 : (int)rint((ftmp-_cropMin)*_nRange/_scaleDelta);
}

} // namepsace lofar
} // namepsace pelican
