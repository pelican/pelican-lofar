#ifndef TRIGGEROUTPUT_H
#define TRIGGEROUTPUT_H
#include <QString>
#include <QMap>
#include <QPair>
#include <QtNetwork/QHostAddress>
#include <vector>

#include "pelican/output/AbstractOutputStream.h"
class QUdpSocket;
// class QIODevice;

/**
 * @file TriggerOutput.h
 */

namespace pelican {
class DataBlob;

namespace lofar {
class DedispersionDataAnalysis;
// class DedispersedTimeSeriesF32;

/**
 * @class TriggerOutput
 *  
 * @brief Converts a DataBlob containing event information into a UDP trigger message.
 *
 * @details Configuration options : 
@verbatim
 <TriggerOutput>
   <receiver host="somehost" port="someport" />
 </TriggerOutput>
@endverbatim
 * 
 */

class TriggerOutput : public AbstractOutputStream
{
    public:
        TriggerOutput( const ConfigNode& configNode  );
        ~TriggerOutput();

        // add a receiver for UDP message
        void addReceiver(const QString& host, quint16 port );
	
        // add a local logfile 
        void addFile( const QString& filename );
	
    protected:
        virtual void sendStream(const QString& streamName, const DataBlob* dataBlob);

    private:
        void _convertToTrigger_FRATS( const DedispersionDataAnalysis* ); 
	void _send(QString message);

    private:
        QMap<QUdpSocket*, QPair<QHostAddress,quint16> > _sockets;
        QList<QIODevice*> _devices;
        /*
	std::vector<float>  _dmValues;
	*/
	QString _idroot;
	QString _format;
	int _min_events;
	float _snr_threshold;
	int _message_counter;
	QString _beamRA, _beamDec, _beamAz, _beamAlt;
        QString _station;
	QString _cfreq_MHz;
	void _set_RA_Dec(void);
	
};

PELICAN_DECLARE(AbstractOutputStream, TriggerOutput )

} // namespace lofar
} // namespace pelican
#endif // TRIGGEROUTPUT_H 
