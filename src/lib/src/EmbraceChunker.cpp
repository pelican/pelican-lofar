#include "EmbraceChunker.h"
#include <QtNetwork/QAbstractSocket>
#include <iostream>

#include <unistd.h> // for closing the socket
#include <sys/socket.h>
#include <netinet/in.h> // for using hton 
#include <linux/if_ether.h> // for the definition of ETH_P_ALL
#include <fcntl.h> // for bloquing socket
#include <errno.h>
#include <stdlib.h>


namespace pelican {

namespace lofar {


/**
 *@details EmbraceChunker 
 */
EmbraceChunker::EmbraceChunker( const ConfigNode& config )
    : AbstractChunker( config )
{
    _sizeOfFrame = 6974;
    _socketBufferSize = 1000000000;
}

/**
 *@details
 */
EmbraceChunker::~EmbraceChunker()
{
}

QIODevice* EmbraceChunker::newDevice()
{
    QAbstractSocket* lsocket = new QAbstractSocket(QAbstractSocket::UnknownSocketType, 0);

    int sock = socket(PF_INET, SOCK_PACKET, htons(ETH_P_ALL));
    if (sock < 0) {
        std::cerr << "socket : %s\n" << strerror(errno) << std::endl;
    }
    if (setsockopt(sock, SOL_SOCKET, SO_RCVBUFFORCE, (char *)&_socketBufferSize, sizeof(unsigned long long)) < 0) {
        std::cerr << "sock opt : %s\n" << strerror(errno) << std::endl;
    }
    lsocket->setSocketDescriptor(sock);
    return lsocket;
}

void EmbraceChunker::next(QIODevice* device)
{
    QAbstractSocket* socket = static_cast<QAbstractSocket*>(device);
    WritableData writableData = getDataStorage(_sizeOfFrame);
    int readTotal = 0;
    if (writableData.isValid() ) {
        do {
            int read = socket->read((char*)writableData.ptr() + readTotal*sizeof(char) , _sizeOfFrame);
            if( read == -1 ) { 
                std::cerr << "read error";  
                // TODO - clean up and annul writableData
            }
            readTotal += read;
        } while ( readTotal < _sizeOfFrame );
    }
    else {
        // TODO dump the data here
        std::cerr << "EmbraceChunker: "
                "WritableData is not valid!" << endl;
    }
}
} // namespace lofar
} // namespace pelican
