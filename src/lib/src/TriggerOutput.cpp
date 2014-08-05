#include "TriggerOutput.h"

#include "pelican/utility/ConfigNode.h"
#include "pelican/data/DataBlob.h"

#include "DedispersionDataAnalysis.h"
/*
#include "SpectrumDataSet.h"
*/

#include <QtNetwork/QUdpSocket>
#include <QtNetwork/QHostAddress>
//#include <QtNetwork/QTcpSocket>
#include <QtCore/QIODevice>
#include <QtCore/QVector>
#include <QtCore/QFile>
#include <QtCore/QTimer>
#include <QtCore/QStringList>
#include <QtCore/QDateTime>
#include <QDebug>
#include <QtGui/QApplication>
#include <iostream>


namespace pelican {
  
namespace ampp {


/**
 *@details TriggerOutput
 */
TriggerOutput::TriggerOutput(const ConfigNode& configNode)
    : AbstractOutputStream( configNode )
{
    // Initliase server connections
    int port = configNode.getOption("Receiver", "port").toInt();
    QString host = configNode.getOption("Receiver", "host");
    std::cout << "Receiver added " << port <<  " Host " << host.toStdString();
    if( host != "" && port !=0 )
    {
        addReceiver( host, port );
    }
    
    // initialize trigger information
    _format = configNode.getOption("Trigger", "format"); 

   // initialise file connections
    QString logfile = configNode.getOption("Logfile", "name");
    if( logfile != "" )
    {
        addFile( logfile );
    }

    // initialize identifier
    _station = configNode.getOption("Telescope", "stationID");
    _idroot = QString("%1-%2%3");
    _idroot = _idroot.arg( QDateTime::currentDateTime().toString("yyyyMMddhhmm") );
    _idroot = _idroot.arg( configNode.getOption("Telescope", "beamID") , 2 , '0');

    // initialize beam direction and frequency
    _beamRA = configNode.getOption("RADec", "rajd");
    _beamDec = configNode.getOption("RADec", "decjd");
    _beamAz = configNode.getOption("AltAz", "az");
    _beamAlt = configNode.getOption("AltAz", "alt");
    _cfreq_MHz = configNode.getOption("frequencyChannel1", "MHz");
    _snr_threshold = configNode.getOption("Threshold", "snr").toFloat();
    _min_events = configNode.getOption("Threshold", "events").toInt();
    _message_counter = 0;
    /*
    // specify dms values to write out
    QString dms = configNode.getOption("DMs", "values", "0");

    // TODO: Process DMs string to extract proper values;
    QStringList dmList = dms.split(",", QString::SkipEmptyParts);
    foreach(QString val, dmList) {
        _dmValues.insert(_dmValues.end(), val.toFloat());
    }
    */
}

/**
 *@details
 */
TriggerOutput::~TriggerOutput()
{
    foreach( QUdpSocket* device, _sockets.keys() )
    {
        delete device;
    }
    
    foreach( QIODevice* device, _devices )
    {
        delete device;
    }
    
}


void TriggerOutput::addFile(const QString& logfile)
{
    QFile* file = new QFile(logfile);
    if( file->open( QIODevice::Append ) ) 
    {
        _devices.append( file );
    }
    /*    
    else if( file->open( QIODevice::WriteOnly ) )
    {
        _devices.append( file );
    }
    */
    else {
        std::cerr << "unable to open file for writing: " << logfile.toStdString() << std::endl;
        delete file;
    }
}

void TriggerOutput::addReceiver(const QString& host, quint16 port )
{
    QUdpSocket* s = new QUdpSocket;
    s->moveToThread(QApplication::instance()->thread()) ;
    _sockets.insert( s, QPair<QHostAddress,quint16>(QHostAddress(host),port) );
    //    _connect( s, host, port );
}

/*
void PumaOutput::_connect(QTcpSocket* tcpSocket , const QString& server, quint16 port)
{
    while ( tcpSocket->state() == QAbstractSocket::UnconnectedState) {
        tcpSocket->connectToHost( server, port );
         if (!tcpSocket -> waitForConnected(5000) || tcpSocket->state() == QAbstractSocket::UnconnectedState) {
            std::cerr << "Client could not connect to server:" << server.toStdString() << " port:" << port << std::endl;
            sleep(2);
            continue;
        }
    }
}
*/

void TriggerOutput::sendStream(const QString& /*streamName*/, const DataBlob* dataBlob)
{
    if( _format == "FRATS" ) {
      _convertToTrigger_FRATS( static_cast<const DedispersionDataAnalysis*>(dataBlob) );
    }
    else {
      std::cerr << "Trigger format not recognized / implemented";
    }
}

void TriggerOutput::_convertToTrigger_FRATS( const DedispersionDataAnalysis* data )
{
    DedispersionEvent* emax;
    float SNR, SNRmax;
    QString message;

    float rms = data->getRMS();
    if (data->eventsFound() >= _min_events) {
        foreach( const DedispersionEvent& e, data->events() ) {
	    if ( (SNR = e.mfValue()/(rms * sqrt(e.mfBinning()))) > SNRmax ) {
	        SNRmax = SNR;
		emax = const_cast<DedispersionEvent*>(&e);
	    }
            
	}
	if (emax && SNRmax >= _snr_threshold) {
	    // set RA and Dec if necessary
	    if (_beamRA == "") _set_RA_Dec();
	    // building trigger message
	    message = "artemis:%1##ID:%2##cFreqMHz:%3##beamRA:%4##beamDec:%5##eTime:%6##eDM:%7##eSNR:%8";
	    message = message.arg( _station );
	    message = message.arg( _idroot.arg( QString::number( ++_message_counter ) , 4 , '0' ) );
	    message = message.arg( _cfreq_MHz , 10 , '0' );
	    message = message.arg( _beamRA , 10 , '0' );
	    message = message.arg( _beamDec , 9 , '0' );
	    message = message.arg( QString::number(emax->getTime(),'f',9) , 20, '0' );
	    message = message.arg( QString::number(emax->dm(),'f',3) , 9, '0' );
	    message = message.arg( QString::number(SNRmax,'f',2) , 6 , '0' );
	    _send( message );
            qDebug() << "Trigger sent and log written " << _beamDec;
	}
    }
}
  
void TriggerOutput::_set_RA_Dec(void)
{
  std::cerr << "No RA/Dec found, conversion not yet implemented";
}
  
void TriggerOutput::_send(QString message)
{
  // send out the data to all required devices
  foreach( QUdpSocket* sock, _sockets.keys() )
    {
      sock->writeDatagram( message.toAscii() , message.size()*sizeof(char), _sockets[sock].first, _sockets[sock].second );
    }
  std::cout << "Datagram written" << std::endl;
  // log the trigger message
  message.append("\n");
  foreach( QIODevice* device, _devices )
    {
      device->write( message.toAscii(), message.size()*sizeof(char) );
    }
}
  
} // namespace ampp
} // namespace pelican
