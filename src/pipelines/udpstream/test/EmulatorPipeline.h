#ifndef EMULATORPIPELINE_H
#define EMULATORPIPELINE_H

#include <QString>

#include "pelican/core/AbstractPipeline.h"
#include "pelican/utility/PelicanTimeRecorder.h"

/**
 * @file EmulatorPipeline.h
 */

namespace pelican {

namespace ampp {
class LofarEmulatorDataSim;

/**
 * @class EmulatorPipeline
 *  
 * @brief
 *    Pipeline to monitor the emulator
 * @details
 * 
 */

class EmulatorPipeline : public AbstractPipeline
{
    public:
        EmulatorPipeline( const QString& streamName , LofarEmulatorDataSim* emulator);

        ~EmulatorPipeline();
        void init();
        void run(QHash<QString, DataBlob*>& remoteData);
        // number of times run called
        unsigned long count() const;

    private:
        unsigned long _runCount;
        QString _dataStream;
        LofarEmulatorDataSim* _emulator;
        PelicanTimeRecorder _recorder;

};

} // namespace ampp
} // namespace pelican
#endif // EMULATORPIPELINE_H 
