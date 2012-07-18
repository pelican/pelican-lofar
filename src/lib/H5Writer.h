#ifndef H5WRITER_H
#define H5WRITER_H

/**
 * @file H5Writer.h
 */

#include "pelican/output/AbstractOutputStream.h"
#include "pelican/utility/ConfigNode.h"
#include "pelican/data/DataBlob.h"

#include <QtCore/QDataStream>
#include <QtCore/QFile>
#include <fstream>

namespace pelican {
namespace lofar {

class SpectrumDataSetStokes;

/**
 * @class H5Writer
 *
 * @brief
 *
 * @details
 *
 */

class H5Writer : public AbstractOutputStream
{
    // lifted from lofar RTCP/Interface/include/Interface/Parset.h
    enum StokesType { STOKES_I = 0, STOKES_IQUV, STOKES_XXYY, INVALID_STOKES = -1 };

    public:
    /// Constructor
        H5Writer( const ConfigNode& config );

    /// Destructor
        ~H5Writer();

    /// File path
        QString filepath() { return _filePath; }

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

PELICAN_DECLARE(AbstractOutputStream, H5Writer)

} // namespace lofar
} // namespace pelican

#endif // H5WRITER_H
