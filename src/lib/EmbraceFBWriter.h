#ifndef EMBRACEFBWRITER_H
#define EMBRACEFBWRITER_H

/**
 * @file EmbraceFBWriter.h
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
 * @class EmbraceFBWriter
 *
 * @brief
 *
 * @details
 *
 */

class EmbraceFBWriter : public AbstractOutputStream
{
    public:
	/// Constructor
        EmbraceFBWriter( const ConfigNode& config );

	/// Destructor
        ~EmbraceFBWriter();

	/// File path
        QString filepath() { return _filepath; }

    protected:
        virtual void sendStream(const QString& streamName, const DataBlob* dataBlob);

    private:
        // Header helpers
        void WriteString(QString string);
        void WriteString(QString string1, QString string2);
        void WriteInt(QString name, int value);
        void WriteFloat(QString name, float value);
        void WriteDouble(QString name, double value);
        void WriteDouble(QString name, double value1, double value2);
        void WriteLong(QString name, long value);
        void writeHeader(SpectrumDataSetStokes* stokes);
        // Data helpers
    protected:
        // buffer and write data in blocks
        void _write(char*,size_t);
        inline void _float2int(const float *f, int *i);

    private:
        bool              _first;
        QString           _filepath;
        std::ofstream     _file1,_file2, _file;
        std::vector<char>  _buffer;
        QString       _sourceNameX, _raStringX, _decStringX;
        QString       _sourceNameY, _raStringY, _decStringY;
        float         _fch1, _foff, _tsamp, _refdm, _clock, _raX, _decX, _raY, _decY;
        float         _cropMin, _cropMax, _scaleDelta, _nRange; // for scaling floats to lesser values
        int           _nchans, _nTotalSubbands;
        int           _buffSize, _cur;
        unsigned int  _nRawPols, _nChannels, _nSubbands, _integration, _nPols;
        unsigned int  _nSubbandsToStore, _topsubband, _lbahba, _site, _machine;
        unsigned int  _nBits;
};

PELICAN_DECLARE(AbstractOutputStream, EmbraceFBWriter)

} // namespace lofar
} // namespace pelican

#endif // EMBRACEFBWRITER_H
