#ifndef GPU_JOB_H
#define GPU_JOB_H

#include <QtCore/QList>
#include <QtCore/QMutex>
#include <QtCore/QWaitCondition>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include "GPU_MemoryMap.h"

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
 *    Specifies a Job to run, its input and output data
 * @details
 * 
 */

class GPU_Job
{
    public:
        typedef enum{ None, Queued, Running, Finished, Failed } JobStatus;

    public:
        GPU_Job();
        ~GPU_Job();
        GPU_Job( const GPU_Job& job );

        const GPU_Job& operator=(const GPU_Job& job);

        void addKernel( GPU_Kernel* kernel );
        const QList<GPU_Kernel*>& kernels() { return _kernels; };
        inline void setStatus( const JobStatus& status ) { _status = status; };
        void setAsRunning();
        inline JobStatus status() const { return _status; };
        const std::string& error() const { return _errorMsg; }
        void setError( const std::string& msg ) { _errorMsg = msg; }
        void emitFinished();
        void wait() const;
        void addCallBack( const boost::function0<void>& fn ) { _callbacks.append(fn); };
        const QList<boost::function0<void> >& callBacks() const { return _callbacks; };
        void reset();

    private:
        std::string _errorMsg;
        QList<GPU_Kernel*> _kernels;
        // status variables
        bool _processing;
        mutable QMutex _mutex;
        mutable QWaitCondition* _waitCondition;
        QList<boost::function0<void> > _callbacks;
        JobStatus _status;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_JOB_H
