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

        // Data helpers
    protected:
        // buffer and write data in blocks
        void _write(char*,size_t);

    private:
        QString       _filepath;
        std::ofstream _file;
	std::vector<char> _buffer;
        float         _fch1, _foff, _tsamp, _refdm;
        int           _nchans, _nTotalSubbands;
	int _buffSize;
        unsigned	   _nPols, _nChannels, _nSubbands, _clock, _integration ;
        unsigned	   _nSubbandsToStore;
	int _cur;
};

PELICAN_DECLARE(AbstractOutputStream, SigprocStokesWriter)

  

  }
}

#endif // SIGPROCSTOKESWRITER_H
