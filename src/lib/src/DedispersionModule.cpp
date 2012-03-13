#include <QDebug>
#include <QList>
#include "DedispersionModule.h"
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include "GPU_MemoryMap.h"
#include "DedispersionBuffer.h"
#include "WeightedSpectrumDataSet.h"
#include "GPU_NVidiaConfiguration.h"
#include "GPU_Job.h"
#include "GPU_Kernel.h"
#include "GPU_Param.h"
#include <fstream>

extern "C" void cacheDedisperseLoop( float *outbuff, long outbufSize, float *buff, float mstartdm,
                                     float mdmstep, int tdms, const int numSamples,
                                     const float* dmShift, const int i_maxshift,
                                     const int i_nchans );


namespace pelican {

namespace lofar {

/**
 *@details DedispersionModule 
 */
DedispersionModule::DedispersionModule( const ConfigNode& config )
    : AsyncronousModule(config)
{
    // Get configuration options
    //unsigned int nChannels = config.getOption("outputChannelsPerSubband", "value", "512").toUInt();
    _numSamplesBuffer = config.getOption("sampleNumber", "value", "512").toUInt();
    _tdms = config.getOption("dedispersionSamples", "value", "1984").toUInt();
    _dmStep = config.getOption("dedispersionStepSize", "value", "0.0").toFloat();
    _dmLow = config.getOption("dedispersionMinimum", "value", "0").toFloat();
    if( _dmLow < 0.0 ) { _dmLow = 0.0; }
    _fch1 = config.getOption("frequencyChannel1", "value", "0.0").toDouble();
    _foff = config.getOption("channelBandwidth", "value", "1.0").toDouble();
    _tsamp = config.getOption("sampleTime", "value", "0.0").toDouble();
    if( _tsamp == 0.000 ) { throw QString("DedispersionModule: must specify a sampleTime"); }
    if( _foff >= 0 ) { throw QString("DedispersionModule: channelBandwidth must be a negative number"); }
    if( _fch1 == 0 ) { throw QString("DedispersionModule: frequencyChannel1 must be a positve number"); }

    unsigned int maxBuffers = config.getOption("numberOfBuffers", "value", "3").toUInt();
    if( maxBuffers < 1 ) throw(QString("DedispersionModule: Must have at least one buffer"));

    // calculate required parameters
    _i_nSamples = GPU_MemoryMap( &_numSamplesBuffer, sizeof(int) );

    // setup the data buffers and objects required for each job
    for( unsigned int i=0; i < maxBuffers; ++i ) {
        _buffersList.append( new DedispersionBuffer(_numSamplesBuffer, 1) );
        GPU_Job tmp;
        _jobs.append( tmp );
    }
    _jobBuffer.reset( &_jobs );
    _buffers.reset( &_buffersList );
    _currentBuffer = _buffers.next();
}

/**
 *@details
 */
DedispersionModule::~DedispersionModule()
{
    _cleanBuffers();
    foreach( DedispersionKernel* k, _kernelList ) {
        delete k;
    }
}

void DedispersionModule::_cleanBuffers() {
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
        Q_ASSERT( (int)maxSamples > _maxshift );
        std::cout << "resize: maxSamples = " << maxSamples << std::endl;
        std::cout << "resize: dmLow = " << _dmLow << std::endl;
        std::cout << "resize: mshift = " << _dmLow + _dmStep * (_tdms - 1) * _dmshifts[nChannels - 1] << std::endl;
        std::cout << "resize: dmStep = " << _dmStep << std::endl;
        std::cout << "resize: tdms = " << _tdms << std::endl;
        std::cout << "resize: foff = " << _foff << std::endl;
        std::cout << "resize: fch1 = " << _fch1 << std::endl;
        std::cout << "resize: maxShift = " << _maxshift << std::endl;
        std::cout << "resize: tsamp = " << _tsamp << std::endl;
        std::cout << "resize: nchans= " << nChannels << std::endl;
        // reset kernels
        foreach( DedispersionKernel* k, _kernelList ) {
            delete k;
        }
        _kernelList.clear();
        _f_dmshifts = GPU_MemoryMap( _dmshifts );
        for( unsigned int i=0; i < maxBuffers; ++i ) {
            DedispersionKernel* kernel = new DedispersionKernel( _dmLow, _dmStep, _tsamp, _tdms,
                                                                 _nChannels, _maxshift );
            _kernelList.append( kernel ); 
            kernel->addConstant( _f_dmshifts );
            kernel->addConstant( _i_nSamples );
        }
        _kernels.reset( &_kernelList );
    }
}

void DedispersionModule::dedisperse( DataBlob* incoming,
                                 LockingPtrContainer<DedispersionSpectra>* dataOut ) {
    dedisperse( dynamic_cast<WeightedSpectrumDataSet*>(incoming), dataOut );
}

void DedispersionModule::dedisperse( WeightedSpectrumDataSet* weightedData, 
                        LockingPtrContainer<DedispersionSpectra>* dataOut )
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

    lock( weightedData );
    // --------- copy spectrum data to buffer -----------------
    SpectrumDataSet<float>* streamData = weightedData->dataSet();
    resize( streamData ); // ensure we have buffers scaled appropriately

    unsigned int sampleNumber = 0; // marker to indicate the number of samples succesfully 
                                   // transferred to the buffer from the Datablob
    unsigned int maxSamples = streamData->nTimeBlocks();
    do {
        if( _currentBuffer->addSamples( weightedData, &sampleNumber ) == 0 ) {
            //(*_currentBuffer)->dump("input.data");
            DedispersionBuffer* next = _buffers.next();
            lock( _currentBuffer->copy( next, _maxshift ) );
            DedispersionSpectra* dedispersionObj = dataOut->next();
            _dedispersionBuffer.insert(dedispersionObj, dataOut); // record lock manager for dd data
            dedisperse( _currentBuffer, dedispersionObj );
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
    //unsigned int outputSize;
    //size_t outputSize = nsamp * _tdms * sizeof(float);
    GPU_MemoryMap out( dataOut->data() );
    kernelPtr->addOutputMap( out );
    kernelPtr->addInputMap( GPU_MemoryMap( buffer->getData() ) );
    //qDebug() << "input buffer:" << buffer->getData();
    job->addKernel( kernelPtr );
    job->addCallBack( boost::bind( &DedispersionModule::gpuJobFinished, this, job, buffer, kernelPtr, dataOut ) );
    submit( job );
}

void DedispersionModule::gpuJobFinished( GPU_Job* job, DedispersionBuffer* buffer, DedispersionKernel* kernel, DedispersionSpectra* dataOut ) {
     Q_ASSERT( job->status() != GPU_Job::Failed ||
        std::cerr << "DedispersionModule: " << job->error() << std::endl
     );
     dataOut->setInputDataBlobs( buffer->inputDataBlobs() );
     kernel->reset();
     _kernels.unlock( kernel ); // give up the kernel
     job->reset();
     _jobBuffer.unlock(job); // return the job to the pool, ready for the next
     _buffers.unlock( buffer ); // give up the buffer
     exportData( dataOut );  // send out the finished data product to our customers
}

void DedispersionModule::exportComplete( DataBlob* datablob ) {
    // unlock Spectrum Data blobs
    DedispersionSpectra* data = static_cast<DedispersionSpectra*>(datablob);
    foreach( WeightedSpectrumDataSet* d, data->inputDataBlobs() ) {
        unlock( d );
    }
    // unlock the dedispersion datablob
    _dedispersionBuffer[data]->unlock(data);
}

DedispersionModule::DedispersionKernel::DedispersionKernel( float start, float step, float tsamp, float tdms , unsigned nChans, unsigned maxshift )
   : _startdm( start ), _dmstep( step ), _tsamp(tsamp), _tdms(tdms), _nChans(nChans),
     _maxshift(maxshift)
{
}

void DedispersionModule::DedispersionKernel::run(const QList<GPU_Param*>& param ) {
     Q_ASSERT( param.size() == 4 );
     //cache_dedisperse_loop( float *outbuff, float *buff, float mstartdm, float mdmstep )
    unsigned nsamples = param[2]->value<int>();
//std::cout << " maxShift =" << _maxshift << std::endl;
//std::cout << " nchans =" << _nChans << std::endl;
//std::cout << " tsamp =" << _tsamp << std::endl;
//std::cout << " input buffer size =" << param[0]->size() << std::endl;
//std::cout << " dmShift size =" << param[1]->size() << std::endl;
//std::cout << " nSamples =" << nsamples << std::endl;
     cacheDedisperseLoop( (float*)param[3]->device() , param[3]->size(),
                          (float*)param[0]->device(), (_startdm/_tsamp),
                          (_dmstep/_tsamp), _tdms, nsamples,
                          (const float*)param[1]->device(),
                          _maxshift,
                          _nChans
                        );
}

void DedispersionModule::DedispersionKernel::reset() {
    _config.clearInputMaps();
    _config.clearOutputMaps();
}

} // namespace lofar
} // namespace pelican
