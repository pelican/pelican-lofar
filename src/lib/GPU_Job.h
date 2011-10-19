#ifndef GPU_JOB_H
#define GPU_JOB_H

#include <QtCore/QList>

/**
 * @file GPU_Job.h
 */

namespace pelican {

namespace lofar {
class GPU_Kernel;


/**
 * @class GPU_Job
 *  
 * @brief
 *    Specifies a Job to run
 * @details
 * 
 */

class GPU_Job
{
    public:
        GPU_Job();
        ~GPU_Job();
        void memcpy( char* start, unsigned long size) { _mem=start; _memSize=size; };
        void addKernel( const GPU_Kernel& kernel );
        const QList<const GPU_Kernel*>& kernels() { return _kernels; };
        unsigned long memSize() const { return _memSize; };
        char* memStart() const { return _mem; };

    private:
        QList<const GPU_Kernel*> _kernels;
        unsigned long _memSize;
        char* _mem;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_JOB_H
