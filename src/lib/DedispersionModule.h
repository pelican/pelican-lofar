#ifndef DEDISPERSIONMODULE_H
#define DEDISPERSIONMODULE_H


#include "boost/function.hpp"
#include <QVector>
#include <QList>
#include "pelican/core/AbstractModule.h"
#include "pelican/utility/LockingCircularBuffer.hpp"
#include "LockingContainer.hpp"
#include "LockingPtrContainer.hpp"
#include "DedispersionSpectra.h"
#include "AsyncronousModule.h"
#include "GPU_Kernel.h"
#include "GPU_MemoryMap.h"
#include "SpectrumDataSet.h"

/**
 * @file DedispersionModule.h
 */

namespace pelican {
class DataBlob;

namespace lofar {
class WeightedSpectrumDataSet;
class GPU_DataMapping;
class GPU_Job;
class GPU_Param;
class DedispersionBuffer;
class LockingBuffer;

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
              unsigned _tdms;
              unsigned _nChans;
              unsigned _maxshift;
           public:
              DedispersionKernel( float, float, float, float, unsigned, unsigned );
              void run(const QList<GPU_Param*>& param );
              void reset();
        };

    public:
        DedispersionModule( const ConfigNode& config );
        ~DedispersionModule();

        /// wait for all asyncronous tasks that have been launched to complete
        void waitForJobCompletion();

        /// processing the incoming data, generating a new DedispersedSpectra in
        /// the process
        //DedispersionSpectra* dedisperse( DataBlob* incoming );
        /// processing the incoming data, filling the provided DedispersedSpectra
        void dedisperse( DataBlob* incoming, 
                                 LockingPtrContainer<DedispersionSpectra>* dataOut );
        void dedisperse( WeightedSpectrumDataSet* incoming,
                                 LockingPtrContainer<DedispersionSpectra>* dataOut );

        /// clean up after gpu task is finished
        void gpuJobFinished( GPU_Job* job,  
                             DedispersionBuffer* buffer, 
                             DedispersionKernel* kernel,
                             DedispersionSpectra* dataOut );

        /// clean up after asyncronous task is finished
        void exportComplete( DataBlob* data );

        /// resize the buffers if necessary to accomodate the provided streamData
        //  If a resize is required any existing data in the buffers
        //  will be lost.
        void resize( const SpectrumDataSet<float>* streamData );

        /// deprecated
        int maxshift() const { return _maxshift; }

     protected:
        void dedisperse( DedispersionBuffer* buffer, DedispersionSpectra* dataOut );
        void _cleanBuffers();

    private:
        QVector<float> _means;
        QVector<float> _rmss;
        unsigned _tdms; 
        double _tsamp; // the time delta that is represented by each sample
        unsigned _numSamplesBuffer;
        float _dmStep;
        float _dmLow;
        double _fch1;
        double _foff;
        QList<DedispersionBuffer*> _buffersList;
        LockingPtrContainer<DedispersionBuffer> _buffers;
        QList<GPU_Job> _jobs;
        LockingContainer<GPU_Job> _jobBuffer; // collection of job objects
        int _maxshift; // number of samples to overlap between processes
        int _nChannels; // number of Channels per sample
        DedispersionBuffer* _currentBuffer;
        GPU_MemoryMap _i_nSamples;
        GPU_MemoryMap _f_dmshifts;
        QVector<float> _dmshifts;

        QList<DedispersionKernel*> _kernelList; // collection of pre-configured kernels
        LockingPtrContainer<DedispersionKernel> _kernels;
        QHash< DataBlob*, LockingPtrContainer<DedispersionSpectra>* > _dedispersionBuffer;
};

PELICAN_DECLARE_MODULE(DedispersionModule)
} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONMODULE_H 
