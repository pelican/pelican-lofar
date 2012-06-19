#ifndef PIPELINEWRAPPER_H
#define PIPELINEWRAPPER_H


#include "pelican/core/AbstractPipeline.h"
#include "pelican/core/PipelineApplication.h"
#include "timer.h"
#include <QDebug>

/**
 * @file PipelineWrapper.h
 */

namespace pelican {
class AbstractPipeline;
class PipelineApplication;

namespace lofar {

/**
 * @class PipelineWrapper
 *  
 * @brief
 *     Wraps a pipeline to provide timing information and
 *     a run once functionality
 * @details
 *     To invoke the pipeline use the PelicanApplication::start() method.
 *     After running the wrapped pipeline, the stop() method will be called
 * 
 */


class PipelineWrapper 
{
    public:
        PipelineWrapper( PipelineApplication* app );
        virtual ~PipelineWrapper() {};
        void setIterations( unsigned );
        virtual void run( QHash<QString, DataBlob*>& data ) = 0;

    protected:
        PipelineApplication* _app_;
        unsigned _pipelineWrapperIterationCount;
        TimerData _runTime_;
};

template<class AbstractPipelineType>
class PipelineWrapperSpecialisation : public AbstractPipelineType,
                                      public PipelineWrapper
{
    public:
        PipelineWrapperSpecialisation( AbstractPipelineType* pipeline, PipelineApplication* app ) 
            : AbstractPipelineType(*pipeline), PipelineWrapper( app )
            {}
        virtual ~PipelineWrapperSpecialisation() {};
        void run( QHash<QString, DataBlob*>& data ) {
            qDebug() << "PipelineWrapperc::run" << data.keys() << " iteration:" << _pipelineWrapperIterationCount;
            timerStart(&_runTime_);
            this->AbstractPipelineType::run(data);
            timerUpdate(&_runTime_);
            if( ! --_pipelineWrapperIterationCount  ) {
                _app_->stop();
                timerReport(&_runTime_, "Pipeline Time: run()");
            }
        }


};

} // namespace lofar
} // namespace pelican
#endif // PIPELINEWRAPPER_H 
