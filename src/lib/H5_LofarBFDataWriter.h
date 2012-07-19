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
 *
 * @details
 *
 */

class H5_LofarBFDataWriter : public AbstractOutputStream
{
    // lifted from lofar RTCP/Interface/include/Interface/Parset.h
    enum StokesType { STOKES_I = 0, STOKES_IQUV, STOKES_XXYY, INVALID_STOKES = -1 };

    public:
    /// Constructor
        H5_LofarBFDataWriter( const ConfigNode& config );

    /// Destructor
        ~H5_LofarBFDataWriter();

    /// File path
        QString filepath() { return _filePath; }

        /// the name of the currently opened raw data file
        const QString& rawFilename() const { return _rawFilename; }
        const QString& metaFilename() const { return _h5Filename; }

    protected:
        void _writeHeader(SpectrumDataSetStokes* stokes);
        virtual void sendStream(const QString& streamName, const DataBlob* dataBlob);

    private:
        // Header helpers
        void _updateHeader();

    protected:
        // buffer and write data in blocks
        void _write(char*,size_t);
        inline void _float2int(const float *f, int *i);

    private:
        QString           _filePath;
        QString           _observationID;
        std::ofstream     _file;
        long              _fileBegin;
        DAL::BF_File*     _bfFile;
        QString           _rawFilename;
        QString           _h5Filename;
        int _beamNr;
        int _sapNr;
        std::vector<ssize_t> _maxdims;
        std::vector<char>  _buffer;
        QString       _sourceName, _raString, _decString, _telescope;
        float         _fch1, _foff, _tsamp, _refdm, _clock, _ra, _dec;
        float         _cropMin, _cropMax, _scaleDelta, _nRange; // for scaling floats to lesser values
        int           _nchans, _nTotalSubbands;
        int           _buffSize, _cur;
        unsigned int  _nRawPols, _nChannels, _nSubbands, _integration, _nPols;
        unsigned int  _nSubbandsToStore, _topsubband, _lbahba, _site, _machine;
        unsigned int  _nBits,_datatype;
};

PELICAN_DECLARE(AbstractOutputStream, H5_LofarBFDataWriter)

} // namespace lofar
} // namespace pelican

#endif // H5_LOFARBFDATAWRITER_H
