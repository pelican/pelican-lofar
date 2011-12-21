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
        _buffersList.append( new DedispersionBuffer(bufferSize) );
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
    // clean up the data buffers
    foreach( DedispersionBuffer* b, _buffersList ) {
        delete b;
    }
    foreach( DedispersionKernel* k, _kernelList ) {
        delete k;
    }
}

DedispersedTimeSeries<float>* DedispersionModule::dedisperse( DataBlob* incoming ) {
    return dedisperse( static_cast<WeightedSpectrumDataSet*>(incoming), new DedispersedTimeSeries<float> );
}

DedispersedTimeSeries<float>* DedispersionModule::dedisperse( WeightedSpectrumDataSet* weightedData, DedispersedTimeSeries<float>* dataOut )
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
    unsigned int sampleNumber = 0; // marker to indicate the number of samples succesfully 
                                   // transferred to the buffer from the Datablob
    while( (*_currentBuffer)->addSamples( weightedData, &sampleNumber ) ) {
        dedisperse( _currentBuffer, dataOut );
        DedispersionBuffer** next = _buffers.next();
        (*_currentBuffer)->copy( *next, _maxshift );
        sampleNumber = 0;
        _currentBuffer = next;
    }
    return dataOut;
}

void DedispersionModule::dedisperse( DedispersionBuffer** buffer, DedispersedTimeSeries<float>* dataOut )
{
    // prepare the output data datablob
    dataOut->resize( (*buffer)->numSamples() );
    // Set up a job for the GPU processing kernel
    GPU_Job* job = _jobBuffer.next();
    DedispersionKernel** kernelPtr = _kernels.next();
    //unsigned int outputSize;
    //GPU_MemoryMap out( &dataOut, outputSize * sizeof(float) );
    //config.addOutputMap( out );
    GPU_MemoryMap in( *buffer, (*buffer)->size() );
    (*kernelPtr)->addInputMap( GPU_MemoryMap( (*buffer)->getData(), (*buffer)->size()) );
    //job->addKernel( *kernelPtr );
    //job->addCallBack( boost::bind( &DedispersionModule::dataExtract, this, out, dataOut) );
    //job->addCallBack( boost::bind( &DedispersionModule::gpuJobFinished, this, job, buffer, kernelPtr) );
    submit( job );
}

void DedispersionModule::gpuJobFinished( GPU_Job* job, DedispersionBuffer** buffer, DedispersionKernel** kernel ) {
     _buffers.unlock( buffer ); // give up the buffer
     (*kernel)->reset();
     _kernels.unlock( kernel ); // give up the kernel
     job->reset();
     _jobBuffer.unlock(job); // return the job to the pool, ready for the next
     //exportData( data );     // send out the finished data product to our customers
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

void DedispersionModule::DedispersionKernel::run(const QList<void*>& /*param*/) {
     //cudaMemset((float*)param[0], 0, size(param[0]) );
     //cache_dedisperse_loop( float *outbuff, float *buff, float mstartdm, float mdmstep )
     /*
     cacheDedisperseLoop( (float*)param[0] , (float*)param[1], _startdm/_tsamp, _dmstep/_tsamp,
                            (const float*)param[2], (const float*)param[3], (const float*)param[4],
                            (const float*)param[5]
                          );
     */
}

void DedispersionModule::DedispersionKernel::reset() {
    _config.clearInputMaps();
    _config.clearOutputMaps();
}

} // namespace lofar
} // namespace pelican
