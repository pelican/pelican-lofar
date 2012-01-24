#include <QList>
#include "DedispersionModule.h"
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include "GPU_MemoryMap.h"
#include "DedispersedTimeSeries.h"
#include "DedispersionBuffer.h"
#include "WeightedSpectrumDataSet.h"
#include "GPU_NVidiaConfiguration.h"
#include "GPU_Job.h"
#include "GPU_Kernel.h"
#include "GPU_Param.h"

extern "C" void cacheDedisperseLoop( float *outbuff, long outbufSize, float *buff, float mstartdm,
                                     float mdmstep, int tdms, int numSamples,
                                     const float* dmShift,
                                     const int* i_nsamp, const int* i_maxshift,
                                     const int* i_nchans );


namespace pelican {

namespace lofar {

/**
 *@details DedispersionModule 
 */
DedispersionModule::DedispersionModule( const ConfigNode& config )
    : AsyncronousModule(config)
{
    // Get configuration options
    unsigned int nChannels = config.getOption("outputChannelsPerSubband", "value", "512").toUInt();
    unsigned int bufferSize = config.getOption("dedispersionSampleNumber", "value", "512").toUInt();
    unsigned int sampleSize = config.getOption("dedispersionSampleSize", "value", "512").toUInt();
    _tdms = config.getOption("dedispersionParameters", "value", "1984").toUInt();
    float dmLow = config.getOption("dispersionMinimum", "value", "0").toFloat();
    float dmStep = config.getOption("dispersionStepSize", "value", "0.1").toFloat();
    unsigned int dmNumber = config.getOption("dispersionSteps", "value", "100").toUInt();
    float fch1 = config.getOption("frequencyChannel1", "value", "0").toFloat();
    float foff = config.getOption("channelBandwidth", "value", "0").toFloat();

    unsigned int maxBuffers = config.getOption("numberOfBuffers", "value", "2").toUInt();
    if( maxBuffers < 1 ) throw(QString("DedispersionModule: Must have at least one buffer"));

    float tsamp = 0;

    // calculate required parameters
    for ( unsigned int c = 0; c < nChannels; c++) {
        _dmshifts.append(  4148.741601 * ((1.0 / (fch1 + (foff * c)) / (fch1 + (foff * c))) - (1.0 / fch1 / fch1)) );
    }
    _maxshift = ((dmLow + dmStep * (dmNumber - 1)) * _dmshifts[nChannels - 1])/tsamp;
    _i_maxshift = GPU_MemoryMap( &_maxshift, sizeof(int) );
    _i_nsamp = GPU_MemoryMap( &_nsamp, sizeof(int) );
    _i_chans = GPU_MemoryMap( &_nChannels, sizeof(int) );
    _f_dmshifts = GPU_MemoryMap( _dmshifts );

    // setup the data buffers and objects required for each job
    for( unsigned int i=0; i < maxBuffers; ++i ) {
        _buffersList.append( new DedispersionBuffer(bufferSize, sampleSize) );
        GPU_Job tmp;
        _jobs.append( tmp );
        DedispersionKernel* kernel = new DedispersionKernel( dmLow, dmStep, tsamp );
        _kernelList.append( kernel ); 
        kernel->addConstant( _f_dmshifts );
        kernel->addConstant( _i_nsamp );
        kernel->addConstant( _i_maxshift );
        kernel->addConstant( _i_chans );
    }
    _jobBuffer.reset( &_jobs );
    _buffers.reset( &_buffersList );
    _kernels.reset( &_kernelList );
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
    if( sampleSize != (*_currentBuffer)->sampleSize() ) {
        unsigned maxBuffers = _buffersList.size();
        unsigned maxSamples = (*_currentBuffer)->maxSamples();
        _cleanBuffers();
        for( unsigned int i=0; i < maxBuffers; ++i ) {
            _buffersList.append( new DedispersionBuffer(maxSamples, sampleSize) );
        }
        _buffers.reset( &_buffersList );
        _currentBuffer = _buffers.next();
    }
}

DedispersedTimeSeries<float>* DedispersionModule::dedisperse( DataBlob* incoming,
                                 LockingCircularBuffer<DedispersedTimeSeries<float>* >* dataOut ) {
    return dedisperse( dynamic_cast<WeightedSpectrumDataSet*>(incoming), dataOut );
}

DedispersedTimeSeries<float>* DedispersionModule::dedisperse( WeightedSpectrumDataSet* weightedData, 
                        LockingCircularBuffer<DedispersedTimeSeries<float>* >* dataOut )
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
    SpectrumDataSet<float>* streamData = weightedData->dataSet();
    resize( streamData ); // ensure we have buffers scaled appropriately

    unsigned int sampleNumber = 0; // marker to indicate the number of samples succesfully 
                                   // transferred to the buffer from the Datablob
    unsigned int maxSamples = streamData->nTimeBlocks();
    do {
        if( (*_currentBuffer)->addSamples( weightedData, &sampleNumber ) == 0 ) {
            dedisperse( _currentBuffer, dataOut->next() );
            DedispersionBuffer** next = _buffers.next();
            (*_currentBuffer)->copy( *next, _maxshift );
            _currentBuffer = next;
        } 
    }
    while( sampleNumber != maxSamples );
    return dataOut->current();
}

void DedispersionModule::dedisperse( DedispersionBuffer** buffer, DedispersedTimeSeries<float>* dataOut )
{
    // prepare the output data datablob
    unsigned int nsamp = (*buffer)->numSamples();
    dataOut->resize( nsamp );
    // Set up a job for the GPU processing kernel
    GPU_Job* job = _jobBuffer.next();
    DedispersionKernel** kernelPtr = _kernels.next();
    //unsigned int outputSize;
    size_t outputSize = nsamp * _tdms * sizeof(float);
    GPU_MemoryMap out( &dataOut, outputSize );
    (*kernelPtr)->addOutputMap( out );
    GPU_MemoryMap in( *buffer, (*buffer)->size() );
    (*kernelPtr)->addInputMap( GPU_MemoryMap( (*buffer)->getData(), (*buffer)->size()) );
    job->addKernel( *kernelPtr );
    job->addCallBack( boost::bind( &DedispersionModule::gpuJobFinished, this, job, buffer, kernelPtr, dataOut ) );
    submit( job );
}

void DedispersionModule::gpuJobFinished( GPU_Job* job, DedispersionBuffer** buffer, DedispersionKernel** kernel, DedispersedTimeSeries<float>* dataOut ) {
     _buffers.unlock( buffer ); // give up the buffer
     (*kernel)->reset();
     _kernels.unlock( kernel ); // give up the kernel
     job->reset();
     _jobBuffer.unlock(job); // return the job to the pool, ready for the next
     exportData( dataOut );  // send out the finished data product to our customers
}

DedispersedTimeSeries<float>* DedispersionModule::dataExtract( const float* /*gpuData*/ , DedispersedTimeSeries<float>* data )
{
    // copy gpu results into an appropriate datablob
    //unsigned int nsamp = samples / _binsize;
    //unsigned int shift = 0;
    //for (unsigned int dm = 0; dm < ndms; ++dm ) {
        //data  = dedispersedData->samples(totdms + dm);
        //data->resize(nsamp);
        //data->setDmValue( startdm + dm * dmstep );
        //memcpy(data->ptr(), &gpuData[shift], nsamp * sizeof(float));
        //shift += nsamp;
    //}
    return data;
}

DedispersionModule::DedispersionKernel::DedispersionKernel( float start, float step, float tsamp ) 
   : _startdm( start ), _dmstep( step ), _tsamp( tsamp )
{
}

void DedispersionModule::DedispersionKernel::run(const QList<GPU_Param*>& param ) {
     Q_ASSERT( param.size() == 6 );
     //cudaMemset((float*)param[0], 0, param[0]->size() );
     //cache_dedisperse_loop( float *outbuff, float *buff, float mstartdm, float mdmstep )
std::cout << "DedispersionModule::DedispersionKernel::run: " << std::endl;
     cacheDedisperseLoop( (float*)param[0]->device() , param[0]->size(),
                          (float*)param[1]->device(), _startdm,
                          (int)(_startdm/_tsamp), (int)(_dmstep/_tsamp),
                          param[3]->value<int>(),
                          (const float*)param[2]->device(),
                          (const int*)param[3]->device(),
                          (const int*)param[4]->device(),
                          (const int*)param[5]->device()
                        );
}

void DedispersionModule::DedispersionKernel::reset() {
    _config.clearInputMaps();
    _config.clearOutputMaps();
}

} // namespace lofar
} // namespace pelican
