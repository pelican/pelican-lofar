#ifndef DEDISPERSIONMODULE_H
#define DEDISPERSIONMODULE_H


#include "boost/function.hpp"
#include <QVector>
#include <QList>
#include "pelican/core/AbstractModule.h"
#include "LockingContainer.hpp"
#include "DedispersedTimeSeries.h"
#include "AsyncronousModule.h"
#include "GPU_Task.h"
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

        void gpuJobFinished( GPU_Job* job,  DedispersionBuffer** buffer );
        DedispersedTimeSeries<float>* dataExtract( const float* outputData, DedispersedTimeSeries<float>* dataBlob );

     protected:
        void recycleGPUTasks( GPU_Task* task, DataBlob* );
        void setupJob( GPU_Job* job, DataBlob* incoming );
        void dedisperse( DedispersionBuffer** buffer, DedispersedTimeSeries<float>* dataOut );

    private:
        QVector<float> _means;
        QVector<float> _rmss;
        QList<DedispersionBuffer*> _buffersList;
        LockingContainer<DedispersionBuffer*> _buffers;
        unsigned int _nsamp; // number of samples per processing block
        int _maxshift; // number of samples to overlap between processes
        DedispersionBuffer** _currentBuffer;
        GPU_MemoryMap _i_nsamp;
        GPU_MemoryMap _i_chans;
        GPU_MemoryMap _i_maxshift;
};

PELICAN_DECLARE_MODULE(DedispersionModule)
} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONMODULE_H 
