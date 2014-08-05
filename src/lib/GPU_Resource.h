#ifndef GPU_RESOURCE_H
#define GPU_RESOURCE_H
#include <QObject>
#include <QMutex>
#include <QWaitCondition>

/**
 * @file GPU_Resource.h
 */

namespace pelican {
namespace ampp {
class GPU_Job;

/**
 * @class GPU_Resource
 *  
 * @brief
 *    Base class for all GPU Resource Types
 * @details
 * 
 */

class GPU_Resource
{

    public:
        GPU_Resource();
        virtual ~GPU_Resource();
        void exec(GPU_Job*);

    protected:
        virtual void run( GPU_Job* job) = 0;

    private:

};

} // namespace ampp
} // namespace pelican
#endif // GPU_RESOURCE_H 
