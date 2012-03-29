#ifndef LOFARPIPELINETESTER_H
#define LOFARPIPELINETESTER_H
#include "pelican/utility/Config.h"
#include "PipelineWrapper.h"
// Lofar Specific Data Types
#include "SpectrumDataSet.h"
#include "TimeSeriesDataSet.h"

#include <QString>

/**
 * @file LofarPipelineTester.h
 */

namespace pelican {

class PipelineApplication;
class AbstractPipeline;

namespace lofar {
class LofarDataBlobGenerator;

/**
 * @class LofarPipelineTester
 *
 * @brief
 *     Utility class to launch and run a pipeline
 *     with Lofar specific data blobs
 * @details
 * 
 */

class LofarPipelineTester
{
    public:
        template <class AbstractPipelineType>
        LofarPipelineTester( AbstractPipelineType* pipeline, const QString& configXML ) : _pipeline(0)
        {
            _init( configXML );
            _pipeline = new 
                PipelineWrapper<AbstractPipelineType>( pipeline, _app );
            _app->registerPipeline(_pipeline);
        }

        ~LofarPipelineTester();
        void run();

    private: 
        void _init( const QString& );

    private:
        Config _config;
        PipelineApplication* _app;
        AbstractPipeline* _pipeline;
        LofarDataBlobGenerator* _dataGenerator;
};

} // namespace lofar
} // namespace pelican
#endif // LOFARPIPELINETESTER_H 
