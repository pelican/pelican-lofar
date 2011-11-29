#ifndef GPU_TASK_H
#define GPU_TASK_H


#include "AsyncronousTask.h"
#include <boost/function.hpp>

/**
 * @file GPU_Task.h
 */

namespace pelican {
class DataBlob;

namespace lofar {
class GPU_Job;
class GPU_Manager;
class GPU_Job;

/**
 * @class GPU_Task
 *  
 * @brief
 *     A Task that runs on a GPU
 * @details
 * 
 */

class GPU_Task : public AsyncronousTask
{
    Q_OBJECT

    public:
        typedef boost::function2<void, GPU_Job*, DataBlob*> preprocessingFunctorT;
        typedef boost::function1<DataBlob*, const GPU_Job* > postprocessingFunctorT;

    public:
        GPU_Task( GPU_Manager* manager, const preprocessingFunctorT&, 
                  const postprocessingFunctorT& );
        ~GPU_Task();
        DataBlob* runJob( DataBlob* data );

    private:
        GPU_Manager* _manager;
        preprocessingFunctorT _pre;
        postprocessingFunctorT _postProcessingTask;

};

} // namespace lofar
} // namespace pelican
#endif // GPU_TASK_H 
