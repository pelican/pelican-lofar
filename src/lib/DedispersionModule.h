#ifndef DEDISPERSIONMODULE_H
#define DEDISPERSIONMODULE_H


#include "boost/function.hpp"
#include <QVector>
#include <QList>
#include "pelican/core/AbstractModule.h"
#include "LockingContainer.hpp"
#include "DedispersedTimeSeries.h"
#include "AsyncronousModule.h"
#include "GPU_Kernel.h"
#include "GPU_MemoryMap.h"

/**
 * @file DedispersionModule.h
 */

namespace pelican {
class DataBlob;

namespace lofar {
class WeightedSpectrumDataSet;
class GPU_DataMapping;
class GPU_Job;
class DedispersionBuffer;

/**
 * @class DedispersionModule
 *  
 * @brief
 *     Run Dedispersion
 * @details
 *     An Asyncronous dedispersion module
 */

class DedispersionModule : public AsyncronousModule
{
   private: 
        // the nvidia kernel description
        class DedispersionKernel : public GPU_Kernel { 
              float _startdm;
              float _dmstep;
              float _tsamp;
           public:
              DedispersionKernel( float, float, float );
              void run(const QList<void*>& params);
              void reset();
        };

    public:
        DedispersionModule( const ConfigNode& config );
        ~DedispersionModule();
        /// processing the incoming data, generating a new DedispersedTimeSeries in
        /// the process
        DedispersedTimeSeries<float>* dedisperse( DataBlob* incoming );
        /// processing the incoming data, filling the provided DedispersedTimeSeries
        DedispersedTimeSeries<float>* dedisperse( DataBlob* incoming, 
                                                  DedispersedTimeSeries<float>* dataOut );
        DedispersedTimeSeries<float>* dedisperse( WeightedSpectrumDataSet* incoming, DedispersedTimeSeries<float>* dataOut );

        void gpuJobFinished( GPU_Job* job,  DedispersionBuffer** buffer, DedispersionKernel** kernel );
        DedispersedTimeSeries<float>* dataExtract( const float* outputData, DedispersedTimeSeries<float>* dataBlob );

     protected:
        void dedisperse( DedispersionBuffer** buffer, DedispersedTimeSeries<float>* dataOut );

    private:
        QVector<float> _means;
        QVector<float> _rmss;
        QList<DedispersionBuffer*> _buffersList;
        LockingContainer<DedispersionBuffer*> _buffers;
        QList<GPU_Job> _jobs;
        LockingContainer<GPU_Job> _jobBuffer; // collection of job objects
        unsigned int _nsamp; // number of samples per processing block
        int _maxshift; // number of samples to overlap between processes
        int _nChannels; // number of Channels per sample
        DedispersionBuffer** _currentBuffer;
        GPU_MemoryMap _i_nsamp;
        GPU_MemoryMap _i_chans;
        GPU_MemoryMap _i_maxshift;
        GPU_MemoryMap _f_dmshifts;
        QVector<float> _dmshifts;


        QList<DedispersionKernel*> _kernelList; // collection of pre-configured kernels
        LockingContainer<DedispersionKernel*> _kernels;
};

PELICAN_DECLARE_MODULE(DedispersionModule)
} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONMODULE_H 
