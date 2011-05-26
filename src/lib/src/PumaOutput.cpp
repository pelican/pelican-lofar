#include "PumaOutput.h"

#include "pelican/utility/ConfigNode.h"
#include "pelican/data/DataBlob.h"

#include "DedispersedTimeSeries.h"
#include "SpectrumDataSet.h"

#include <QtNetwork/QTcpSocket>
#include <QtCore/QIODevice>
#include <QtCore/QVector>
#include <QtCore/QFile>
#include <QtCore/QTimer>

#include <iostream>


namespace pelican {

namespace lofar {


/**
 *@details PumaOutput
 */
PumaOutput::PumaOutput(const ConfigNode& configNode)
    : AbstractOutputStream( configNode )
{
    // Initliase server connections
    int port = configNode.getOption("connection", "port").toInt();
    QString host = configNode.getOption("connection", "host");
    if( host != "" && port !=0 )
    {
        addServer( host, port );
    }
    QString filename = configNode.getOption("file", "name");
    // initialise file connections
    if( filename != "" )
    {
        addFile( filename );
    }

    // specify dms values to write out
    QString dms = configNode.getOption("DMs", "values", "0");

    // TODO: Process DMs string to extract proper values;
    QStringList dmList = dms.split(",", QString::SkipEmptyParts);
    foreach(QString val, dmList) {
        _dmValues.insert(_dmValues.end(), val.toFloat());
    }
}

/**
 *@details
 */
PumaOutput::~PumaOutput()
{
    foreach( QTcpSocket* device, _sockets.keys() )
    {
        delete device;
    }
    foreach( QIODevice* device, _devices )
    {
        delete device;
    }
}

void PumaOutput::addFile(const QString& filename)
{
    QFile* file = new QFile(filename);
    if( file->open( QIODevice::WriteOnly ) )
    {
        _devices.append( file );
    }
    else {
        std::cerr << "unable to open file for writing: " << filename.toStdString() << std::endl;
    }
}

void PumaOutput::addServer(const QString& host, quint16 port )
{
    QTcpSocket* s = new QTcpSocket;
    _sockets.insert( s, QPair<QString,quint16>(host,port) );
    _connect( s, host, port );
}

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

void PumaOutput::sendStream(const QString& /*streamName*/, const DataBlob* dataBlob)
{
    if( dataBlob->type() == "SpectrumDataSetStokes" )
    {
        _convertToPuma( static_cast<const SpectrumDataSetStokes*>(dataBlob) );
    }
    else if( dataBlob->type() == "DedispersedTimeSeriesF32" )
    {
        _convertToPuma( static_cast<const DedispersedTimeSeriesF32*>(dataBlob) );
    }
}

void PumaOutput::_convertToPuma( const SpectrumDataSetStokes* data )
{
    int blocks = data->nTimeBlocks();
    if ( blocks ) {
	    //
	    // unwrap the datablob and munge channels and subbands together
	    //
	    QVector<float> puma(blocks);
	    puma.fill(0);
	    unsigned int polarisation = 0; // only do one polarisation

	    unsigned int nSubbands = data->nSubbands();
	    unsigned int nChannels = data->nChannels();
	    for (unsigned t = 0; t < data->nTimeBlocks(); ++t) {
		    for (unsigned s = 0; s < nSubbands; ++s) {
			    const float* spectrum = data->spectrumData(t, s, polarisation );
			    for (unsigned int c = 0; c < nChannels ; ++c) {
				    puma[t] = spectrum[c];
			    }
		    }
	    }

	    // send out the data to all required devices
	    foreach( QTcpSocket* sock, _sockets.keys() )
	    {
		    _connect( sock, _sockets[sock].first, _sockets[sock].second );
		    sock->write( (char*)&puma[0], puma.size()*sizeof(float) );
	    }
	    foreach( QIODevice* device, _devices )
	    {
		    device->write( (char*)&puma[0], puma.size()*sizeof(float) );
	    }
    }
}

void PumaOutput::_convertToPuma( const DedispersedTimeSeriesF32* timeData )
{
    if (timeData->nDMs() == 0)
        return;

    for(unsigned i = 0; i < _dmValues.size(); i++) {
	    for(unsigned j = 0; j < timeData->nDMs(); j++) {
		    if (fabs(timeData->samples(j)->dmValue() - _dmValues[i])  < 0.00001) {
			    const DedispersedSeries<float>* series = timeData->samples(j);
			    _send(reinterpret_cast<const char *>(series -> ptr()), series->nSamples() * sizeof(float));
		    }
	    }
    }
}

void PumaOutput::_send(const char* puma, size_t size)
{
    // send out the data to all required devices
    foreach( QTcpSocket* sock, _sockets.keys() )
    {
        _connect( sock, _sockets[sock].first, _sockets[sock].second );
        sock->write( puma, size );
    }
    foreach( QIODevice* device, _devices )
    {
        device->write( puma, size );
    }
}

} // namespace lofar
} // namespace pelican
