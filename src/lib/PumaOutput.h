#ifndef PUMAOUTPUT_H
#define PUMAOUTPUT_H
#include <QString>
#include <QMap>
#include <QPair>
#include <vector>

#include "pelican/output/AbstractOutputStream.h"
class QTcpSocket;
class QIODevice;

/**
 * @file PumaOutput.h
 * @configuration
 * <PumaOutput>
 *   <connection host="somehost" port="someport" />
 *  <file name="somefilename" />
 * </PumaOutput>
 */

namespace pelican {
class DataBlob;

namespace lofar {
class SpectrumDataSetStokes;
class DedispersedTimeSeriesF32;

/**
 * @class PumaOutput
 *  
 * @brief
 *   Converts a Spectrum DataBlob into a TCP stream 
 *   suitable for the puma server
 * @details
 * 
 */

class PumaOutput : public AbstractOutputStream
{
    public:
        PumaOutput( const ConfigNode& configNode  );
        ~PumaOutput();


        // add a puma server to stream to
        void addServer(const QString& host, quint16 port );

        // add a file to which to send data
        void addFile( const QString& filename );

    protected:
        virtual void sendStream(const QString& streamName, const DataBlob* dataBlob);

    private:
        void _convertToPuma( const SpectrumDataSetStokes* );
        void _convertToPuma( const DedispersedTimeSeriesF32* );
        void _connect(QTcpSocket*, const QString&, quint16 port);
	    void _send(const char* puma, size_t size);

    private:
        QMap<QTcpSocket*, QPair<QString,quint16> > _sockets;
        QList<QIODevice*> _devices;
	
	    std::vector<float>  _dmValues;
};

PELICAN_DECLARE(AbstractOutputStream, PumaOutput )

} // namespace lofar
} // namespace pelican
#endif // PUMAOUTPUT_H 
