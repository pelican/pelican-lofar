#ifndef H5_LOFARBFDATAWRITER_H
#define H5_LOFARBFDATAWRITER_H

/**
 * @file H5_LofarBFDataWriter.h
 */

#include "pelican/output/AbstractOutputStream.h"
#include "pelican/utility/ConfigNode.h"
#include "pelican/data/DataBlob.h"

#include <QtCore/QDataStream>
#include <QtCore/QFile>
#include <QtCore/QVector>
#include <QtCore/QList>
#include <fstream>

namespace DAL {
  class BF_File;
}
namespace pelican {
namespace lofar {

class SpectrumDataSetStokes;

/**
 * @class H5_LofarBFDataWriter
 *
 * @brief
 *    OutputStreamer that genererates Lofar Beamformed Format
 *    H5 files
 * @details
 *    <H5_LofarBFDataWriter>
 *        <file filepath="dir/to/save/output/files">
 *        <observation id="MyArtemisObservation" />
 *        <LBA_0_or_HBA_1 value="1">
 *            Define which array this refers to
 *        </LBA_0_or_HBA_1>
 *        <clock value="200">
 *            could be 160
 *        </clock>
 *        <fch1 value="150" />
 *        <topSubbandIndex value="150" >
 *            Used to calculate the frequency of the first channel
 *            fch1 - only used if fch1 is not set
 *        </topSubbandIndex>
 *        <params nPolsToWrite="1" >
 *              the number of stokes parameters to write
 *              - each param will be written to a separate
 *              data file.
 *        </params>
 *    </H5_LofarBFDataWriter>
 */

class H5_LofarBFDataWriter : public AbstractOutputStream
{
    // lifted from lofar RTCP/Interface/include/Interface/Parset.h
    enum StokesType { STOKES_I = 0, STOKES_IQUV, STOKES_XXYY, INVALID_STOKES = -1 };

    public:
        H5_LofarBFDataWriter( const ConfigNode& config );
        ~H5_LofarBFDataWriter();

        /// return the file path
        QString filepath() { return _filePath; }

        /// the name of the currently opened raw data file
        const QString rawFilename( int polarisation ) const { 
                if( polarisation > (int)_nPols ) return "";
                return _rawFilename[polarisation]; 
        }
        const QString metaFilename( int polarisation ) const { 
                if( polarisation > (int)_nPols ) return "";
                return _h5Filename[polarisation]; 
        }

    protected:
        void _writeHeader(SpectrumDataSetStokes* stokes);
        virtual void sendStream(const QString& streamName, const DataBlob* dataBlob);

    private:
        // Header helpers
        void _updateHeader( int polarisation );

    protected:
        inline void _float2int(const float *f, int *i);
        void _setChannels( unsigned n );
        void _setPolsToWrite(unsigned p);

    private:
        QString           _filePath;
        QString           _observationID;
        QVector<std::ofstream*>    _file;
        QVector<long>              _fileBegin;
        QVector<DAL::BF_File*>     _bfFiles;
        QVector<QString>           _rawFilename;
        QVector<QString>           _h5Filename;
        int _beamNr;
        int _sapNr;
        std::vector<ssize_t> _maxdims;
        QString       _sourceName, _raString, _decString, _telescope;
        float         _fch1, _foff, _tsamp, _refdm, _clock, _ra, _dec;
        float         _cropMin, _cropMax, _scaleDelta, _nRange; // for scaling floats to lesser values
        int           _nchans, _nTotalSubbands;
        unsigned int  _nRawPols, _nChannels, _nSubbands, _integration, _nPols;
        unsigned int  _nSubbandsToStore, _topsubband, _lbahba, _site, _machine;
        unsigned int  _nBits;
};

PELICAN_DECLARE(AbstractOutputStream, H5_LofarBFDataWriter)

} // namespace lofar
} // namespace pelican

#endif // H5_LOFARBFDATAWRITER_H
