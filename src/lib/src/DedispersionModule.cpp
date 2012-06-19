#include <QDebug>
#include <QList>
#include "DedispersionModule.h"
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include "GPU_MemoryMap.h"
#include "DedispersionBuffer.h"
#include "WeightedSpectrumDataSet.h"
#include "GPU_Job.h"
#include "GPU_Kernel.h"
#include "GPU_Param.h"
#include "GPU_NVidia.h"
#include "GPU_Manager.h"
#include <fstream>

extern "C" void cacheDedisperseLoop( float *outbuff, long outbufSize, float *buff, float mstartdm,
                                     float mdmstep, int tdms, const int numSamples,
                                     const float* dmShift, const int i_maxshift,
                                     const int i_nchans );


namespace pelican {

namespace lofar {

/**
 *@details DedispersionModule 
 * Example configuration:
 * <DedispersionModule>
 *    <sampleNumber value="512">
 *       The total number of time samples to dedisperse at once
 *    </sampleNumber>
 *    <sampleTime seconds="0.3">
 *       The time represented by each time sample (in seconds)
 *    </sampleNumber>
 *    <frequencyChannel1 MHz="150.0">
 *       The frequency of the first channel (highest freq)
 *    </frequencyChannel1>
 *    <channelBandwidth MHz="-0.03">
 *       The width of each frequency channel. Must be -ve such
 *       that frequencyChannel1 + nChannels*channelBandwidth =
 *       lowest frequency.
 *    </channelBandwidth>
 * </DedispersionModule>
 */
DedispersionModule::DedispersionModule( const ConfigNode& config )
    : AsyncronousModule(config)
{
    // Get configuration options
    //unsigned int nChannels = config.getOption("outputChannelsPerSubband", "value", "512").toUInt();
    _numSamplesBuffer = config.getOption("sampleNumber", "value", "512").toUInt();
    _tdms = config.getOption("dedispersionSamples", "value", "1984").toUInt();
    _dmStep = config.getOption("dedispersionStepSize", "value", "0.0").toFloat();
    _dmLow = config.getOption("dedispersionMinimum", "value", "0.0").toFloat();
    if( _dmLow < 0.0 ) { _dmLow = 0.0; }
    _fch1 = config.getOption("frequencyChannel1", "MHz", "0.0").toDouble();
    _foff = config.getOption("channelBandwidth", "MHz", "1.0").toDouble();
    _tsamp = config.getOption("sampleTime", "seconds", "0.0").toDouble();
    if( _tsamp == 0.000 ) { throw QString("DedispersionModule: must specify a sampleTime"); }
    if( _foff >= 0 ) { throw QString("DedispersionModule: channelBandwidth must be a negative number"); }
    if( _fch1 == 0 ) { throw QString("DedispersionModule: frequencyChannel1 must be a positve number"); }

    unsigned int maxBuffers = config.getOption("numberOfBuffers", "value", "2").toUInt();
    if( maxBuffers < 1 ) throw(QString("DedispersionModule: Must have at least one buffer"));

    // setup the data buffers and objects required for each job
    for( unsigned int i=0; i < maxBuffers; ++i ) {
        _buffersList.append( new DedispersionBuffer(_numSamplesBuffer, 1) );
        GPU_Job tmp;
        _jobs.append( tmp );
        DedispersionSpectra tmp2;
        _dedispersionData.append( tmp2 );
    }
    _jobBuffer.reset( &_jobs );
    _buffers.reset( &_buffersList );
    _dedispersionDataBuffer.reset( &_dedispersionData );
    _currentBuffer = _buffers.next();
}

/**
 *@details
 */
DedispersionModule::~DedispersionModule()
{
    waitForJobCompletion();
    _cleanBuffers();
}

void DedispersionModule::waitForJobCompletion() {
    while( ! ( _kernels.allAvailable() && _jobBuffer.allAvailable() ) ) {
        usleep(10);
    }
    AsyncronousModule::waitForJobCompletion();
}

void DedispersionModule::_cleanBuffers() {
    // clean up kernels
    foreach( DedispersionKernel* k, _kernelList ) {
        delete k;
    }
    _kernelList.clear();
    // clean up the data buffers memory
    foreach( DedispersionBuffer* b, _buffersList ) {
        delete b;
    }
    _buffersList.clear();
}

void DedispersionModule::resize( const SpectrumDataSet<float>* streamData ) {
    
    unsigned int nChannels = streamData->nChannels();
    unsigned int nSubbands = streamData->nSubbands();
    unsigned int nPolarisations = streamData->nPolarisations();
    unsigned sampleSize = nSubbands * nChannels * nPolarisations;
    if( sampleSize != _currentBuffer->sampleSize() ) {
        unsigned maxBuffers = _buffersList.size();
        unsigned maxSamples = _currentBuffer->maxSamples();
        waitForJobCompletion();
        _cleanBuffers();
        for( unsigned int i=0; i < maxBuffers; ++i ) {
            _buffersList.append( new DedispersionBuffer(maxSamples, sampleSize) );
        }
        _buffers.reset( &_buffersList );
        _currentBuffer = _buffers.next();

        _nChannels = nChannels * nSubbands;
        std::cout << "resize: nChannels = " << _nChannels << std::endl;
        // calculate dispersion measure shifts
        _dmshifts.clear();
        for ( int c = 0; c < _nChannels; ++c ) {
            _dmshifts.append(  4148.741601 * ((1.0 / (_fch1 + (_foff * c)) / 
                               (_fch1 + (_foff * c))) - (1.0 / _fch1 / _fch1)) );
        }
        _maxshift = ((_dmLow + _dmStep * (_tdms - 1)) * _dmshifts[_nChannels - 1])/_tsamp;
        std::cout << "resize: maxSamples = " << maxSamples << std::endl;
        std::cout << "resize: dmLow = " << _dmLow << std::endl;
        std::cout << "resize: mshift = " << _dmLow + _dmStep * (_tdms - 1) * _dmshifts[_nChannels - 1] << std::endl;
        std::cout << "resize: dmStep = " << _dmStep << std::endl;
        std::cout << "resize: tdms = " << _tdms << std::endl;
        std::cout << "resize: foff = " << _foff << std::endl;
        std::cout << "resize: fch1 = " << _fch1 << std::endl;
        std::cout << "resize: maxShift = " << _maxshift << std::endl;
        std::cout << "resize: tsamp = " << _tsamp << std::endl;
        std::cout << "resize: blob nChannels= " << nChannels << std::endl;
        std::cout << "resize: nTimeBlocks= " << streamData->nTimeBlocks() << std::endl;
        if( (int)maxSamples <= _maxshift ) {
            throw QString("DedispersionModule: maxshift requirements (%1) are bigger"
                          " than the number of samples (%2)").arg(_maxshift).arg(maxSamples);
        }
        // reset kernels
        for( unsigned int i=0; i < maxBuffers; ++i ) {
            DedispersionKernel* kernel = new DedispersionKernel( _dmLow, _dmStep,
                                _tsamp, _tdms,
                                _nChannels, _maxshift, _numSamplesBuffer );
            _kernelList.append( kernel ); 
            kernel->setDMShift( _dmshifts );
        }
        _kernels.reset( &_kernelList );
    }
}

void DedispersionModule::dedisperse( DataBlob* incoming )
{
    dedisperse( dynamic_cast<WeightedSpectrumDataSet*>(incoming) );
}

void DedispersionModule::dedisperse( WeightedSpectrumDataSet* weightedData )
{
    // transfer weighted data to host memory buffer
    //
    // --------- copy data statitistics -----------------
    // Index is the place to write the mean and rms given the current chunk. It 
    // takes into account the fact that for buffers after the first, the first 
    // values are just copied from the maxshift part of the previous buffers
    //int index = ( _counter == 0 ) ? _nsamp/nTimeBlocks 
    //    : _nsamp / nTimeBlocks + floor( (float)_maxshift/(float) nTimeBlocks);

    //_means[_counter % _stages][index] = blobMean * nChannels;
    //_rmss[_counter % _stages][index] = blobRMS * nChannels;

    // --------- copy spectrum data to buffer -----------------
    SpectrumDataSetStokes* streamData = 
                static_cast<SpectrumDataSetStokes*>(weightedData->dataSet());
    lock( streamData );
    resize( streamData ); // ensure we have buffers scaled appropriately

    unsigned int sampleNumber = 0; // marker to indicate the number of samples succesfully 
                                   // transferred to the buffer from the Datablob
    unsigned int maxSamples = streamData->nTimeBlocks();
    do {
        if( _currentBuffer->addSamples( streamData, &sampleNumber ) == 0 ) {
            //(*_currentBuffer)->dump("input.data");
            DedispersionBuffer* next = _buffers.next();
            next->clear();
            lock( _currentBuffer->copy( next, _maxshift ) );
            // ensure lock is maintianed for the next buffer
            // if not already marked by the maxshift copy
            if( sampleNumber != maxSamples && lockNumber( streamData ) <=1 )
                lock( streamData );
            dedisperse( _currentBuffer, _dedispersionDataBuffer.next() );
            _currentBuffer = next;
        }
    }
    while( sampleNumber != maxSamples );
}

void DedispersionModule::dedisperse( DedispersionBuffer* buffer, DedispersionSpectra* dataOut )
{
    // prepare the output data datablob
    unsigned int nsamp = buffer->numSamples() - _maxshift;
    dataOut->resize( nsamp, _tdms, _dmLow, _dmStep );
    // Set up a job for the GPU processing kernel
    GPU_Job* job = _jobBuffer.next();
    DedispersionKernel* kernelPtr = _kernels.next();
    kernelPtr->setOutputBuffer( dataOut->data() );
    kernelPtr->setInputBuffer( buffer->getData(),
                   boost::bind( &DedispersionModule::gpuDataUploaded, this, buffer ) );
    job->addKernel( kernelPtr );
    job->addCallBack( boost::bind( &DedispersionModule::gpuJobFinished, this, job, kernelPtr, dataOut ) );
    dataOut->setInputDataBlobs( buffer->inputDataBlobs() );
    submit( job );
    //std::cout << "dedispersionModule: current jobs = " << gpuManager()->jobsQueued() << std::endl;
}

void DedispersionModule::gpuJobFinished( GPU_Job* job, DedispersionKernel* kernel, DedispersionSpectra* dataOut ) {
     _kernels.unlock( kernel ); // give up the kernel
     if( job->status() != GPU_Job::Failed ) {
         job->reset();
         _jobBuffer.unlock(job); // return the job to the pool, ready for the next
         exportData( dataOut );  // send out the finished data product to our customers
     } else {
         std::cerr << "DedispersionModule: " << job->error() << std::endl;
         job->reset();
         _jobBuffer.unlock(job); // return the job to the pool, ready for the next
         exportCancel( dataOut );
     }
}

void DedispersionModule::gpuDataUploaded( DedispersionBuffer* buffer ) {
    _buffers.unlock(buffer);
}

void DedispersionModule::exportComplete( DataBlob* datablob ) {
    // unlock Spectrum Data blobs
    DedispersionSpectra* data = static_cast<DedispersionSpectra*>(datablob);
    foreach( SpectrumDataSetStokes* d, data->inputDataBlobs() ) {
        unlock( d );
    }
    // unlock the dedispersion datablob
    _dedispersionDataBuffer.unlock(data);
}

DedispersionModule::DedispersionKernel::DedispersionKernel( float start, float step, float tsamp, float tdms , unsigned nChans, unsigned maxshift, unsigned nsamples )
   : _startdm( start ), _dmstep( step ), _tsamp(tsamp), _tdms(tdms), _nChans(nChans),
     _maxshift(maxshift), _nsamples(nsamples)
{
}

void DedispersionModule::DedispersionKernel::setDMShift( QVector<float>& buffer ) {
    _dmShift = GPU_MemoryMap(buffer);
}

void DedispersionModule::DedispersionKernel::cleanUp() {
    _inputBuffer.runCallBacks();
}

void DedispersionModule::DedispersionKernel::setOutputBuffer( QVector<float>& buffer )
{
    _outputBuffer = GPU_MemoryMap(buffer);
}

void DedispersionModule::DedispersionKernel::setInputBuffer( QVector<float>& buffer, GPU_MemoryMap::CallBackT callback ) {
    _inputBuffer = GPU_MemoryMap(buffer);
    _inputBuffer.addCallBack( callback );
}

void DedispersionModule::DedispersionKernel::run( GPU_NVidia& gpu ) {
     //cache_dedisperse_loop( float *outbuff, float *buff, float mstartdm, float mdmstep )
//std::cout << " maxShift =" << _maxshift << std::endl;
//std::cout << " nchans =" << _nChans << std::endl;
//std::cout << " tsamp =" << _tsamp << std::endl;
//std::cout << " input buffer (" << gpu.devicePtr(_inputBuffer) << " ) size=" << _inputBuffer.size() << std::endl;
//std::cout << " output buffer (" << gpu.devicePtr(_outputBuffer) << ") size=" << _outputBuffer.size() << std::endl;
//std::cout << " dmShift size =" << _dmShift.size() << std::endl;
//std::cout << " nSamples =" << _nsamples << std::endl;
     cacheDedisperseLoop( (float*)gpu.devicePtr(_outputBuffer) , _outputBuffer.size(),
                          (float*)gpu.devicePtr(_inputBuffer), (_startdm/_tsamp),
                          (_dmstep/_tsamp), _tdms, _nsamples,
                          (const float*)gpu.devicePtr(_dmShift),
                          _maxshift,
                          _nChans
                        );
}

} // namespace lofar
} // namespace pelican
