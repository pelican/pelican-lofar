#include "PumaOutput.h"
#include <QtNetwork/QTcpSocket>
#include <QIODevice>
#include <QVector>
#include <QFile>
#include <iostream>

#include "pelican/utility/ConfigNode.h"
#include "pelican/data/DataBlob.h"

#include "SpectrumDataSet.h"


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

void PumaOutput::send(const QString& /*streamName*/, const DataBlob* dataBlob)
{
    if( dataBlob->type() == "SpectrumDataSetStokes" )
    {
        _convertToPuma( static_cast<const SpectrumDataSetStokes*>(dataBlob) );
    }
}

void PumaOutput::_convertToPuma( const SpectrumDataSetStokes* data )
{
    //
    // unwrap the datablob and munge channels and subbands together
    //
    QVector<float> puma;
    unsigned int polarisation = 0; // only do one polarisation
    
    unsigned int nSubbands = data->nSubbands();
    unsigned int nChannels = data->nChannels();
    for (unsigned t = 0; t < data->nTimeBlocks(); ++t) {
        for (unsigned s = 0; s < nSubbands; ++s) {
            const float* spectrum = data->spectrumData(t, s, polarisation );
            for (unsigned int c = 0; c < nChannels ; ++c) {
                puma[t] += spectrum[c];
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

} // namespace lofar
} // namespace pelican
