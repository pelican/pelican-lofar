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
#include "timer.h"

#ifdef CUDA_FOUND
/**
 * @file DedispersionModule.h
 */

namespace pelican {
class DataBlob;

namespace ampp {
class WeightedSpectrumDataSet;
class GPU_DataMapping;
class GPU_Job;
class GPU_Param;
class GPU_NVidia;
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
              unsigned _totalNumberOfChannels;
              unsigned _nChans;
              unsigned _maxshift;
              unsigned _nsamples;
              GPU_MemoryMapOutput _outputBuffer;
              GPU_MemoryMap _inputBuffer;
              GPU_MemoryMapConst _dmShift;

           public:
              DedispersionKernel( float, float, float, float, unsigned, unsigned, unsigned );
              void setDMShift( QVector<float>& );
              void setOutputBuffer( std::vector<float>& );
              void setInputBuffer( std::vector<float>&, GPU_MemoryMap::CallBackT );
              void run( GPU_NVidia& );
              void cleanUp();
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
        void dedisperse( DataBlob* incoming );
        void dedisperse( WeightedSpectrumDataSet* incoming );

        /// clean up after gpu task is finished
        void gpuJobFinished( GPU_Job* job,
                             DedispersionKernel* kernel,
                             DedispersionSpectra* dataOut );
        /// return input buffers for reuse as soon as data uploaded to the GPU
        void gpuDataUploaded( DedispersionBuffer* );

        /// clean up after asyncronous task is finished
        void exportComplete( DataBlob* data );

        /// resize the buffers if necessary to accomodate the provided streamData
        //  If a resize is required any existing data in the buffers
        //  will be lost.
        void resize( const SpectrumDataSet<float>* streamData );

        /// return the number of samples collected before dedispersion
        unsigned numberOfSamples() const { return _numSamplesBuffer; }

        /// deprecated
        int maxshift() const { return _maxshift; }

     protected:
        void dedisperse( DedispersionBuffer* buffer, DedispersionSpectra* dataOut );
        void _cleanBuffers();

    private:
        bool _invert;
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
        // number of samples remaining between the ones dedispersed and nsamples-maxshift
        int _remainingSamples;
        int _nChannels; // number of Channels per sample
        DedispersionBuffer* _currentBuffer;
        std::vector<float> _noiseTemplate;
        QVector<float> _dmshifts;

        QVector<SpectrumDataSetStokes*> _blobs;

        QList<DedispersionSpectra> _dedispersionData; // data products for async tasks
        LockingContainer<DedispersionSpectra> _dedispersionDataBuffer;

        QList<DedispersionKernel*> _kernelList; // collection of pre-configured kernels
        LockingPtrContainer<DedispersionKernel> _kernels;

        // Timers
        DEFINE_TIMER( _copyTimer )
        DEFINE_TIMER( _bufferTimer )
        DEFINE_TIMER( _launchTimer )
        DEFINE_TIMER( _dedisperseTimer )

};

PELICAN_DECLARE_MODULE(DedispersionModule)
} // namespace ampp
} // namespace pelican
#endif // CUDA_FOUND
#endif // DEDISPERSIONMODULE_H
