#ifndef LOFARPIPELINETESTER_H
#define LOFARPIPELINETESTER_H

#include <QString>

/**
 * @file LofarPipelineTester.h
 */

namespace pelican {

class PipelineApplication;
class AbstractPipeline;

namespace lofar {
class PipelineWrapper;

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
        LofarPipelineTester( AbstractPipeline* pipeline, const QString& configXML );
        ~LofarPipelineTester();
        void run();

    private:
        PipelineApplication* _app;
        PipelineWrapper* _pipeline;
};

} // namespace lofar
} // namespace pelican
#endif // LOFARPIPELINETESTER_H 
