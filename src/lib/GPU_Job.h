#ifndef GPU_JOB_H
#define GPU_JOB_H

#include <QtCore/QList>
#include <boost/shared_ptr.hpp>

/**
 * @file GPU_Job.h
 */

namespace pelican {

namespace lofar {
class GPU_Kernel;
class GPU_MemoryMap;


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
        GPU_Job();
        ~GPU_Job();
        void addKernel( const GPU_Kernel& kernel );
        const QList<const GPU_Kernel*>& kernels() { return _kernels; };
        void setInputMap( const boost::shared_ptr<GPU_MemoryMap>& map );
        void setOutputMap( const boost::shared_ptr<GPU_MemoryMap>& map );
        const QList<boost::shared_ptr<GPU_MemoryMap> >& inputMemoryMaps() const { return _inputMaps; };
        const QList<boost::shared_ptr<GPU_MemoryMap> >& outputMemoryMaps() const { return _outputMaps; };

    private:
        QList<const GPU_Kernel*> _kernels;
        QList<boost::shared_ptr<GPU_MemoryMap> > _outputMaps;
        QList<boost::shared_ptr<GPU_MemoryMap> > _inputMaps;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_JOB_H
