#include <dal/lofar/BF_File.h>
#include <dal/dal_version.h>
#include <dal/lofar/Coordinates.h>

#include "SpectrumDataSet.h"
#include "H5Writer.h"

#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <complex>
#include <vector>

namespace pelican {
namespace lofar {


// Constructor
// TODO: For now we write in 32-bit format...
H5Writer::H5Writer(const ConfigNode& configNode )
  : AbstractOutputStream(configNode), _first(true)
{
    _filePath = configNode.getOption("file", "filepath");
    _fileName = configNode.getOption("file", "filename", "LofarBFData");
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
      _foff = -_clock / (_nRawPols * _nTotalSubbands) / float(_nChannels);
    }
    else{
      _foff = configNode.getOption("foff", "value", "1.0").toFloat();
    }

    // Observation sampling time
    if( configNode.getOption("tsamp", "value" ) == "" ){ 
      _tsamp = (_nRawPols * _nTotalSubbands) * _nChannels * _integration / _clock/ 1e6;
    }
    else{
      _tsamp=configNode.getOption("tsamp", "value", "1.0").toFloat();
    }

    // Number of polarisations to write out, 1 for total power or 4
    // for stokes and complex voltages
    _nPols=configNode.getOption("params", "nPolsToWrite", "1").toUInt();

    // Number of total numbers
    _nchans=_nChannels * _nSubbands;

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

void H5Writer::_writeHeader(SpectrumDataSetStokes* stokes){
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
      QString h5Basename = _fileName + QString("_S0%1.h5").arg(i);
      QString h5Filename = _filePath + "/" + h5Basename;

      //-------------- File  -----------------
      DAL::BF_File* file = new DAL::BF_File( h5Filename.toStdString(), DAL::BF_File::CREATE);

      // Common Attributes
      std::vector<std::string> stationList; stationList.push_back(_telescope.toStdString());
      file->groupType().value = "Root";
      file->fileName().value = h5Basename.toStdString();
      file->fileType().value = "bf";
      file->telescope().value = _telescope.toStdString();
      file->observer().value = "unknown";
      file->observationNofStations().value = 1;
      file->observationStationsList().value = stationList;
      // TODO file->pipelineName().value = _currentPipelineName;
      file->pipelineVersion().value = ""; // TODO
      file->docName() .value   = "ICD 3: Beam-Formed Data";
      file->docVersion().value = "2.04.27";
      file->notes().value      = "";
      file->createOfflineOnline().value = "Online";
      file->BFFormat().value   = "TAB";
      file->BFVersion().value  = QString("Artemis H5Writer using DAL %1 and HDF5 %2")
                                      .arg(DAL::get_lib_version().c_str())
                                      .arg(DAL::get_dal_hdf5_version().c_str())
                                      .toStdString();

      // Observation Times
      //file->observationStartUTC().value = toUTC(_Time);
      file->observationStartMJD().value = _mjdStamp;
      //file.observationStartTAI().value = toTAI(_startTime);
      
      //  -- Telescope Settings --
      file->clockFrequencyUnit().value = "MHz";
      file->clockFrequency().value = _clock;
      file->observationNofBitsPerSample().value = _nBits;
      file->bandwidth().value = _nSubbands * _foff;
      file->bandwidthUnit().value = "MHz";
      //file->totalIntegrationTime().value = nrBlocks * _integration;
      
      //-------------- subarray pointing  -----------------
      file->nofSubArrayPointings().value = 1;
      int sapNr = 0;
      DAL::BF_SubArrayPointing sap = file->subArrayPointing(sapNr);
      sap.create();
      sap.groupType().value = "SubArrayPointing";
      //sap.expTimeStartUTC().value = toUTC(_startTime);
      //sap.expTimeStartMJD().value = toMJD(_startTime);
      //sap.expTimeStartTAI().value = toTAI(_startTime);
      //sap.expTimeEndUTC().value = toUTC(stopTime);
      //sap.expTimeEndMJD().value = toMJD(stopTime);
      //sap.expTimeEndTAI().value = toTAI(stopTime);
      //sap.totalIntegrationTime().value = parset.beamDuration(sapNr);
      sap.totalIntegrationTimeUnit().value = "s";
      sap.nofBeams().value = 1;

      //-------------- Beam -----------------
      int beamNr = 0;
      DAL::BF_BeamGroup beam = sap.beam(beamNr);
      beam.create();
      beam.groupType().value = "Beam";
      beam.nofStations().value = 1;
      beam.stationsList().value = stationList;

      std::vector<std::string> mytargets;
      mytargets.push_back(_sourceName.toStdString() );  // 1 target only
      beam.targets().value=mytargets;
      beam.nofStokes().value = 1;
      beam.channelsPerSubband().value = _nChannels;
      beam.channelWidthUnit()  .value = "MHz";

      std::vector<std::string> stokesVars;
      int stokesType=STOKES_IQUV;
      int stokesNr=0; // only write the I part
      switch(stokesType) {
        case STOKES_I:
          stokesVars.push_back("I");
          break;

        case STOKES_IQUV:
          stokesVars.push_back("I");
          stokesVars.push_back("Q");
          stokesVars.push_back("U");
          stokesVars.push_back("V");
          break;

        case STOKES_XXYY:
          stokesVars.push_back("Xr");
          stokesVars.push_back("Xi");
          stokesVars.push_back("Yr");
          stokesVars.push_back("Yi");
          break;

        default:
          throw(QString("H5Writer:  INVALID_STOKES"));
          return;
      }
      beam.complexVoltage().value = stokesType;
      std::vector<std::string> stokesComponents(1, stokesVars[stokesNr]);

      // Coordinates within Beam
      DAL::CoordinatesGroup coord = beam.coordinates();
      coord.create();
      coord.groupType().value = "Coordinates";
      coord.nofCoordinates().value = 2;
      coord.nofAxes().value = 2;
      std::vector<std::string> coordinateTypes(2);
      coordinateTypes[0] = "Time"; // or TimeCoord ?
      coordinateTypes[1] = "Spectral"; // or SpectralCoord ?
      coord.coordinateTypes().value = coordinateTypes;

      std::vector<double> unitvector(1,1.0);
      DAL::TimeCoordinate* timeCoordinate = dynamic_cast<DAL::TimeCoordinate*>(coord.coordinate(0));
      timeCoordinate->create();
      timeCoordinate->groupType()     .value = "TimeCoord";

      timeCoordinate->coordinateType().value = "Time";
      timeCoordinate->storageType()   .value = std::vector<std::string>(1,"Linear");
      timeCoordinate->nofAxes()       .value = 1;
      timeCoordinate->axisNames()     .value = std::vector<std::string>(1,"Time");
      timeCoordinate->axisUnits()     .value = std::vector<std::string>(1,"us");
      // linear coordinates:
      //   referenceValue = offset from starting time, in axisUnits
      //   referencePixel = offset from first sample
      //   increment      = time increment for each sample
      //   pc             = scaling factor (?)
      timeCoordinate->referenceValue().value = 0;
      timeCoordinate->referencePixel().value = 0;
      //timeCoordinate->increment()     .value = parset.sampleDuration() * itsInfo.timeIntFactor;
      timeCoordinate->pc()            .value = unitvector;

      timeCoordinate->axisValuesPixel().value = std::vector<unsigned>(1, 0); // not used
      timeCoordinate->axisValuesWorld().value = std::vector<double>(1, 0.0); // not used

      DAL::SpectralCoordinate* spectralCoordinate = 
                    dynamic_cast<DAL::SpectralCoordinate*>(coord.coordinate(1));
      spectralCoordinate->create();
      spectralCoordinate->groupType()     .value = "SpectralCoord";
      spectralCoordinate->coordinateType().value = "Spectral";
      spectralCoordinate->storageType()   .value = std::vector<std::string>(1,"Tabular");
      spectralCoordinate->nofAxes()       .value = 1;
      spectralCoordinate->axisNames()     .value = std::vector<std::string>(1,"Frequency");
      spectralCoordinate->axisUnits()     .value = std::vector<std::string>(1,"MHz");
      spectralCoordinate->referenceValue().value = 0; // not used
      spectralCoordinate->referencePixel().value = 0; // not used
      spectralCoordinate->increment()     .value = 0; // not used
      spectralCoordinate->pc()            .value = unitvector; // not used
/*
      for( unsigned sb = 0; sb < nrSubbands; ++sb ) {
        const double subbandBeginFreq = beamCenterFrequencies[sb] - 0.5 * subbandBandwidth;

        for(unsigned ch = 0; ch < itsInfo.nrChannels; ch++) {
          spectralPixels.push_back(spectralPixels.size());
          spectralWorld .push_back(subbandBeginFreq + ch * channelBandwidth);
        }
      }

      spectralCoordinate.get()->axisValuesPixel().value = spectralPixels;
      spectralCoordinate.get()->axisValuesWorld().value = spectralWorld;
*/


      // =============== Stokes Data ================
      DAL::BF_StokesDataset stokesDS = beam.stokes(stokesNr);
      std::vector<ssize_t> dims(2), maxdims(2);

      dims[0] = 1; //itsNrSamples * nrBlocks;
      dims[1] = _nChannels;

      maxdims[0] = -1;
      maxdims[1] = _nChannels; //itsNrChannels;

      QString rawBasename = _fileName + ".raw";
      QString rawFilename = _filePath + "/" + rawBasename;
      _file.open(rawFilename.toStdString().c_str(),
                        std::ios::out | std::ios::binary);

      stokesDS.create(dims, maxdims, rawBasename.toStdString(),
                      (QSysInfo::ByteOrder == QSysInfo::BigEndian) ? 
                                DAL::BF_StokesDataset::BIG 
                                : DAL::BF_StokesDataset::LITTLE);
      stokesDS.groupType().value = "bfData";
      stokesDS.dataType() .value = "float";

      stokesDS.stokesComponent().value = stokesVars[stokesNr];
      stokesDS.nofChannels().value = std::vector<unsigned>(_nSubbands, _nChannels);
      stokesDS.nofSubbands().value = _nSubbands;
      stokesDS.nofSamples().value = dims[0];
    }
}

// Destructor
H5Writer::~H5Writer()
{
    _updateHeader();
    _file.close();
}

void H5Writer::_updateHeader() {
    // TODO ensure dimesions of binary file are specified in the header
    // meta data
}

// ---------------------------- Header helpers --------------------------
//
// Write data blob to disk
void H5Writer::sendStream(const QString& /*streamName*/, const DataBlob* incoming)
{
    SpectrumDataSetStokes* stokes;
    DataBlob* blob = const_cast<DataBlob*>(incoming);

    if( (stokes = (SpectrumDataSetStokes*) dynamic_cast<SpectrumDataSetStokes*>(blob))){

        if (_first){
            _first = false;
            _writeHeader(stokes);
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
