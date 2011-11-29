#ifndef GPU_RESOURCE_H
#define GPU_RESOURCE_H
#include <QObject>


/**
 * @file GPU_Resource.h
 */

namespace pelican {
namespace lofar {
class GPU_Job;

/**
 * @class GPU_Resource
 *  
 * @brief
 *    Base class for all GPU Resource Types
 * @details
 * 
 */

class GPU_Resource : public QObject
{
    Q_OBJECT

    public:
        GPU_Resource();
        virtual ~GPU_Resource();
        void exec(GPU_Job*);

    protected:
        virtual void run( GPU_Job* job) = 0;

    signals:
        void finished();

    private:
};

} // namespace lofar
} // namespace pelican
#endif // GPU_RESOURCE_H 
