#include <dal/lofar/BF_File.h>
#include <dal/dal_version.h>
#include <dal/lofar/Coordinates.h>

#include "SpectrumDataSet.h"
#include "H5_LofarBFDataWriter.h"
#include "TimeStamp.h"

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
H5_LofarBFDataWriter::H5_LofarBFDataWriter(const ConfigNode& configNode )
  : AbstractOutputStream(configNode), _beamNr(0), _sapNr(0),
        _nChannels(0), _nSubbands(0), _nPols(0)
{
    _filePath = configNode.getOption("file", "filepath", ".");
    _observationID= configNode.getOption("observation", "id", "");

    // By definition for LOFAR RSP boards, the following should not change:
    _nRawPols = configNode.getOption("nRawPolarisations", "value", "2").toUInt();
    _nTotalSubbands = configNode.getOption("totalComplexSubbands", "value", "512").toUInt();

    // Parameters that change for every observation
    _topsubband     = configNode.getOption("topSubbandIndex", "value", "150").toFloat();
    _integration    = configNode.getOption("integrateTimeBins", "value", "1").toUInt();
    _nBits = configNode.getOption("dataBits", "value", "32").toUInt();

    _checkPoint = configNode.getOption("checkPoint", "interval", "2000" ).toUInt();

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
          _fch1 = 100 + _clock / (_nRawPols * _nTotalSubbands) * _topsubband;
        if (_clock == 160)
          _fch1 = 160 + _clock / (_nRawPols * _nTotalSubbands) * _topsubband;
      }
    }
    else{
      _fch1 = configNode.getOption("fch1", "value", "1400.0").toFloat();
    }

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

    _maxdims.resize(2);
    _separateFiles = true; // only separte files/pol currently supported
}

// Destructor
H5_LofarBFDataWriter::~H5_LofarBFDataWriter()
{
    _setPolsToWrite(0);
}

void H5_LofarBFDataWriter::_setChannels( unsigned n ) {
    _nChannels = n;
    _nchans= _nChannels * _nSubbands;
    _tsamp = (_nRawPols * _nTotalSubbands) * _nChannels * _integration / _clock/ 1e6;
    _foff = -_clock / (_nRawPols * _nTotalSubbands) / float(_nChannels);
}

void H5_LofarBFDataWriter::_setPolsToWrite( unsigned n ) {
    _nPols = n;
    // clean up existing
    // n.b only delete _file after calls to 
    // _updateHeader
    for(int i=0; i < (int)_bfFiles.size(); ++i ) {
        _updateHeader( i );
        if( _separateFiles ) {
            if( _file.size() > i ) {
                _file[i]->flush();
                _file[i]->close();
                delete _file[i];
            }
        }
        delete _bfFiles[i];
    }
    if( ! _separateFiles ) delete _file[0];
    // setup for new number of pols
    _bfFiles.resize(n);
    _file.resize(n);
    if( _separateFiles ) {
        for(int i=0; i<_file.size(); ++i ) {
           _file[i] = new std::ofstream;
        }
    }
    else {
        // redirect all ofstreams to the same file
        _file[0] = new std::ofstream;
        for(int i=1; i<_file.size(); ++i ) {
            _file[i] = _file[0];
        }
    }
    _h5Filename.resize(n);
    _rawFilename.resize(n);
    _fileBegin.resize(n);
    _count = 0;
}

void H5_LofarBFDataWriter::_writeHeader(SpectrumDataSetBase* stokes){
    time_t _timeStamp = stokes->getLofarTimestamp();
    TimeStamp timeStamp( _timeStamp );
    double _mjdStamp = timeStamp.mjd();

    // Create a total of _nPols h5 files for writing
    // each with the standard lofar file name format
    // ref doc LOFAR-USG-ICD-005 
    // L<Observation ID>_<Optional Descriptors>_<Filetype>.<Extension>
    // Where optional contains: Sx - Stokes value,  and date/time
    char timestr[22];
    strftime(timestr, sizeof timestr, "D%Y%m%dT%H%M%S.", gmtime(&_timeStamp) );

    // --- remove any old streams
    for (unsigned i=0; i<_nPols; ++i){
      if( _bfFiles[i] ) {
          _updateHeader( i );
          if( _file[i] ) {
              _file[i]->flush();
              _file[i]->close();
          }
          delete _bfFiles[i]; _bfFiles[i] = 0;
      }
    }
    // generate the new headers
    _nSubbands = stokes->nSubbands();
    _setChannels( stokes->nChannels() );
    for (unsigned i=0; i<_nPols; ++i){
      // generate a unique - non existing filename
      QString fileName, h5Basename, tmp;
      int version = -1;
      do {
          fileName = "L" + _observationID + QString("_S%1_").arg(i)
              + timestr + QString("%1Z_bf").arg(++version);
          h5Basename = fileName + QString(".h5");
          tmp = _filePath + "/" + h5Basename;
      } while ( QFile::exists( tmp ) );
      _h5Filename[i] = tmp;

      //-------------- File  -----------------
      DAL::BF_File* bfFile =  new DAL::BF_File( _h5Filename[i].toStdString(), DAL::BF_File::CREATE);
      _bfFiles[i] = bfFile;

      // Common Attributes
      std::vector<std::string> stationList; stationList.push_back(_telescope.toStdString());
      bfFile->groupType().value = "Root";
      bfFile->fileName().value = h5Basename.toStdString();
      bfFile->fileType().value = "bf";
      bfFile->telescope().value = _telescope.toStdString();
      bfFile->observer().value = "unknown";
      bfFile->observationNofStations().value = 1;
      bfFile->observationStationsList().value = stationList;
      // TODO bfFile->pipelineName().value = _currentPipelineName;
      bfFile->pipelineVersion().value = ""; // TODO
      bfFile->docName() .value   = "ICD 3: Beam-Formed Data";
      bfFile->docVersion().value = "2.04.27";
      bfFile->notes().value      = "";
      bfFile->createOfflineOnline().value = "Online";
      bfFile->BFFormat().value   = "TAB";
      bfFile->BFVersion().value  = QString("Artemis H5_LofarBFDataWriter using DAL %1 and HDF5 %2")
                                      .arg(DAL::get_lib_version().c_str())
                                      .arg(DAL::get_dal_hdf5_version().c_str())
                                      .toStdString();

      // Observation Times
      //bfFile->observationStartUTC().value = toUTC(_Time);
      bfFile->observationStartMJD().value = _mjdStamp;
      //bfFile.observationStartTAI().value = toTAI(_startTime);
      
      //  -- Telescope Settings --
      bfFile->clockFrequencyUnit().value = "MHz";
      bfFile->clockFrequency().value = _clock;
      bfFile->observationNofBitsPerSample().value = _nBits;
      bfFile->bandwidth().value = _nSubbands * _foff;
      bfFile->bandwidthUnit().value = "MHz";
      //bfFile->totalIntegrationTime().value = nrBlocks * _integration;
      
      //-------------- subarray pointing  -----------------
      bfFile->nofSubArrayPointings().value = 1;
      DAL::BF_SubArrayPointing sap = bfFile->subArrayPointing(_sapNr);
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
      DAL::BF_BeamGroup beam = sap.beam(_beamNr);
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
      int stokesNr=0; // only write the I part
      switch(_stokesType) {
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
          throw(QString("H5_LofarBFDataWriter:  INVALID_STOKES"));
          return;
      }
      beam.complexVoltage().value = _complexVoltages;
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
      std::vector<ssize_t> dims(2);

      dims[0] = 0; //stokes->nTimeBlocks(); // no data yet
      dims[1] = _nSubbands * _nChannels;

      _maxdims[0] = -1; // no fixed length
      _maxdims[1] = dims[1];

      QString rawBasename = fileName + ".raw";
      _rawFilename[i] = _filePath + "/" + rawBasename;
      _file[i]->open(_rawFilename[i].toStdString().c_str(),
                        std::ios::out | std::ios::binary);
      _fileBegin[i] = _file[i]->tellp(); // store storage loc of first byte to 
                                  // be able to calculate exact data size later
                                  // N.B. using other methods for filesize may only
                                  // be accurate to the nearest disk block/sector
      stokesDS.create(dims, _maxdims, rawBasename.toStdString(),
                      (QSysInfo::ByteOrder == QSysInfo::BigEndian) ? 
                                DAL::BF_StokesDataset::BIG 
                                : DAL::BF_StokesDataset::LITTLE);
      stokesDS.groupType().value = "bfData";
      stokesDS.dataType() .value = "float";

      stokesDS.stokesComponent().value = stokesVars[stokesNr];
      stokesDS.nofChannels().value = std::vector<unsigned>(_nSubbands, _nChannels);
      stokesDS.nofSubbands().value = _nSubbands;
      stokesDS.nofSamples().value = dims[0];
      _bfFiles[i]->flush();
    }
}


void H5_LofarBFDataWriter::_updateHeaders() {
    for(int i=0; i < (int)_bfFiles.size(); ++i ) {
        _updateHeader( i );
    }
}
void H5_LofarBFDataWriter::_updateHeader( int pol ) {
    if( _bfFiles[pol] && _file[pol] ) {
        DAL::BF_SubArrayPointing sap = _bfFiles[pol]->subArrayPointing(_sapNr);
        DAL::BF_BeamGroup beam = sap.beam(_beamNr);
        DAL::BF_StokesDataset stokesDS = beam.stokes(0);
        // update the data dimensions according to the file size
        _maxdims[0] = (_file[pol]->tellp() - _fileBegin[pol])/(_maxdims[1] * _nBits/8);
        stokesDS.resize( _maxdims );
        stokesDS.nofSamples().value = _maxdims[0];
        _bfFiles[pol]->flush();
    }
}

// ---------------------------- Header helpers --------------------------
//
// Write data blob to disk
void H5_LofarBFDataWriter::sendStream(const QString& /*streamName*/, const DataBlob* incoming)
{
    SpectrumDataSetBase* stokes;
    DataBlob* blob = const_cast<DataBlob*>(incoming);

    if( (stokes = (SpectrumDataSetBase*) dynamic_cast<SpectrumDataSetBase*>(blob))){
        unsigned nSubbands = stokes->nSubbands();
        unsigned nChannels = stokes->nChannels();
        unsigned nPolarisations = stokes->nPolarisationComponents();

        // check format of stokes is consistent with the current stream
        // if not then we close the existing stream and open up a new one
        if( _nPols > nPolarisations ) {
            _setPolsToWrite( nPolarisations );
            _writeHeader(stokes);
        } else if( nSubbands != _nSubbands || nChannels != _nChannels ) {
            // start the new stream if the data has changed
            _writeHeader(stokes);
        }
        _writeData( stokes ); // subclass actually writes the data

        // force save to disk at the checkPoint interval
        if( ++_count%_checkPoint == 0 ) {
            // flush the file streams
            for (unsigned p = 0; p < _nPols; ++p) {
               _file[p]->flush();
            }
            _updateHeaders();
        }
    }
}



} // namepsace lofar
} // namepsace pelican
