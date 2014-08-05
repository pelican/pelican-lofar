#include "EmulatorPipeline.h"

#include <iostream>

namespace pelican {

namespace ampp {


/**
 *@details EmulatorPipeline 
 */
EmulatorPipeline::EmulatorPipeline( const QString& streamName,  
                                    LofarEmulatorDataSim* emulator
                                  )
    : AbstractPipeline(), _runCount(0), _dataStream(streamName), _emulator(emulator)
{
}

/**
 *@details
 */
EmulatorPipeline::~EmulatorPipeline()
{
    _recorder.report();
}

void EmulatorPipeline::init() {
    // Request remote data
    std::cout << "EmulatorPipeline::init() : request for " << _dataStream.toStdString() << std::endl;
    requestRemoteData(_dataStream);
    _recorder.setReportInterval(1);
    _recorder.start();
}

void EmulatorPipeline::run(QHash<QString, DataBlob*>& )
{
    _recorder.tick("run");
    ++_runCount;
}

unsigned long EmulatorPipeline::count() const
{
//    std::cout << "runCount=" << _runCount <<std::endl;
    return _runCount;
}
} // namespace ampp
} // namespace pelican
