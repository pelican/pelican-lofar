#ifndef LOFARTESTCLIENT_H
#define LOFARTESTCLIENT_H

#include <QThread>
#include <QString>
#include <QFile>
#include <boost/shared_ptr.hpp>
#include "pelican/core/PipelineApplication.h"

/**
 * @file LofarTestClient.h
 */

namespace pelican {

namespace ampp {
   class LofarEmulatorDataSim;
   class EmulatorPipeline;

/**
 * @class LofarTestClient
 *  
 * @brief
 *    A testing client using the general pipeline infrastructure
 * @details
 * 
 */

class LofarTestClient : public QThread
{
    Q_OBJECT 

    public:
        LofarTestClient( LofarEmulatorDataSim* emulator ,
                         const QFile& configFile,
                         const QString& stream,
                         QObject* parent = 0);
        ~LofarTestClient();

        virtual void run();

        // startup the client and wait
        // for it to get going
        void startup();

        unsigned long count() const; 

    private:
        LofarEmulatorDataSim* _emulator;
        EmulatorPipeline* _pipeline;
        QFile _configFile;
        QString _stream;
        boost::shared_ptr<PipelineApplication> _pApp;
};

} // namespace ampp
} // namespace pelican
#endif // LOFARTESTCLIENT_H 
