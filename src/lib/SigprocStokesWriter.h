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

    public:
        virtual void send(const QString& streamName, const DataBlob* dataBlob);

    private:
        // Header helpers
        void WriteString(QString string);
        void WriteInt(QString name, int value);
        void WriteFloat(QString name, float value);
        void WriteDouble(QString name, double value);
        void WriteLong(QString name, long value);

        // Data helpers

    private:
        QString       _filepath;
        std::ofstream _file;
        float         _fch1, _foff, _tsamp;
        int           _nchans;
        unsigned      _nPols;

};

PELICAN_DECLARE(AbstractOutputStream, SigprocStokesWriter)

}
}

#endif // SIGPROCSTOKESWRITER_H
