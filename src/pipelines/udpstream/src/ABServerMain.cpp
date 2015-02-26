#include "server/PelicanServer.h"
#include "comms/PelicanProtocol.h"
#include "utility/Config.h"

#include "ABChunker.h"

#include <QtCore/QCoreApplication>
#include <iostream>

using namespace pelican;

int main(int argc, char ** argv)
{
    // Create a QCoreApplication.
    QCoreApplication app(argc, argv);

    // Create a Pelican configuration object (this assumes that a Pelican
    // configuration XML file is supplied as the first command line argument)
    if (argc != 2) {
        std::cerr << "Please supply an XML config file." << std::endl;
        return 0;
    }
    QString configFile(argv[1]);
    Config config(configFile);

    try {
        // Create a Pelican server.
        PelicanServer server(&config);

        // Attach the chunker to server.
        server.addStreamChunker("ABChunker");

        // Create a communication protocol object and attach it to the server
        // on port 15000.
        AbstractProtocol *protocol =  new PelicanProtocol;
        server.addProtocol(protocol, 15000);

        // Start the server.
        server.start();

        // When the server is ready enter the QCoreApplication event loop.
        while (!server.isReady()) {}
        return app.exec();
    }
    // Catch any error messages from Pelican.
    catch (const QString& err)
    {
        std::cerr << "Error: " << err.toStdString() << std::endl;
        return 1;
    }
}

