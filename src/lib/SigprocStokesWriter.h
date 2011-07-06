#ifndef SIGPROCSTOKESWRITER_H
#define SIGPROCSTOKESWRITER_H

#include "pelican/output/AbstractOutputStream.h"
#include "pelican/utility/ConfigNode.h"
#include "pelican/data/DataBlob.h"

#include <QtCore/QDataStream>
#include <QtCore/QFile>
#include <fstream>

namespace pelican {
namespace lofar {

class SpectrumDataSetStokes;
class SigprocStokesWriter : public AbstractOutputStream
{
    public:
        SigprocStokesWriter( const ConfigNode& config );
        ~SigprocStokesWriter();
        QString filepath() { return _filepath; }

    protected:
        virtual void sendStream(const QString& streamName, const DataBlob* dataBlob);

    private:
        // Header helpers
        void WriteString(QString string);
        void WriteInt(QString name, int value);
        void WriteFloat(QString name, float value);
        void WriteDouble(QString name, double value);
        void WriteLong(QString name, long value);
        void writeHeader(SpectrumDataSetStokes* stokes);
        // Data helpers
    protected:
        // buffer and write data in blocks
        void _write(char*,size_t);
        void _float2int(const float *f, int n, int b, float min, float max, int *i);

    private:
        bool              _first;
        QString           _filepath;
        std::ofstream     _file;
        std::vector<char>  _buffer;
        QString         _sourceName, _raString, _decString;
        float           _fch1, _foff, _tsamp, _refdm, _clock, _ra, _dec;
        float           _scaleMin, _scaleMax; // for scaling floats to lesser values
        int             _nchans, _nTotalSubbands;
        int             _buffSize, _cur;
        unsigned int    _nRawPols, _nChannels, _nSubbands, _integration, _nPols;
        unsigned int    _nSubbandsToStore, _topsubband, _lbahba, _site, _machine;
        unsigned int    _nBits;
};

PELICAN_DECLARE(AbstractOutputStream, SigprocStokesWriter)

} // namespace lofar
} // namespace pelican

#endif // SIGPROCSTOKESWRITER_H
