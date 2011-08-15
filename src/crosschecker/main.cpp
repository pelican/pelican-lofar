/*
 *
 * Copyright OeRC 2010
 *
 */
#include <QCoreApplication>
#include <QPair>
#include "../lib/LofarPelicanClientApp.h"
#include "pelican/output/ThreadedDataBlobClient.h"
#include <boost/program_options.hpp>
#include "CorrelatingBuffer.h"
#include "CorrelationCheckModule.h"
#include "CorrelatedBufferManager.h"
#include <cstdlib>
#include <iostream>

namespace opts = boost::program_options;
using namespace pelican;
using namespace pelican::lofar;

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);

    Config::TreeAddress address;
    address << Config::NodeId("CrossChecker", "");

    LofarPelicanClientApp pelican(argc, argv, address);
    CorrelatedBufferManager buffers;
    address << Config::NodeId("CheckAlgorithm", "");
    CorrelationCheckModule module( pelican.config(address) );

    bool ok = module.connect(&buffers, 
              SIGNAL(foundCorrelation(QMap<QString, RTMS_Data>)),
              SLOT( run(QMap<QString, RTMS_Data>) )
              );
    Q_ASSERT( ok );
    
    LofarPelicanClientApp::ClientMapContainer_T map=pelican.clients();
    LofarPelicanClientApp::ClientMapContainer_T::iterator i;
    for (i = map.begin(); i != map.end(); ++i) {
        CorrelatingBuffer* buffer = buffers.newBuffer(i.key());
        bool ok = buffer->connect( i.value(), SIGNAL(newData(const Stream&)), 
                         SLOT( newData(const Stream&) ) );
        Q_ASSERT( ok );
    }
    
    try {
        return app.exec();
    }
    catch (const QString& err)
    {
        std::cout << "ERROR: " << err.toStdString() << std::endl;
    }
}
