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
    : AsyncronousModule(config) //, boost::bind( &DedispersionModule::dedisperse, this, _1 ) ) 
{
    // Get configuration options
    unsigned int nChannels = config.getOption("outputChannelsPerSubband", "value", "512").toUInt();
    unsigned int bufferSize = config.getOption("dedispersionSampleNumber", "value", "512").toUInt();
    unsigned int maxBuffers = config.getOption("numberOfBuffers", "value", "2").toUInt();
    if( maxBuffers < 1 ) throw(QString("DedispersionModule: Must have at least one buffer"));
    unsigned int dmLow = config.getOption("dispersionMinimum", "value", "2").toUInt();
    unsigned int dmStep = config.getOption("dispersionStepSize", "value", "2").toUInt();
    unsigned int dmNumber = config.getOption("dispersionSteps", "value", "100").toUInt();
    float fch1 = config.getOption("frequencyChannel1", "value", "0").toFloat();
    float foff = config.getOption("channelBandwidth", "value", "0").toFloat();
    float tsamp = 0;

    // calculate required parameters
    QList<float> dmshifts;
    for ( unsigned int c = 0; c < nChannels; c++) {
        dmshifts.append(  4148.741601 * ((1.0 / (fch1 + (foff * c)) / (fch1 + (foff * c))) - (1.0 / fch1 / fch1)) );
    }
    _maxshift = ((dmLow + dmStep * (dmNumber - 1)) * dmshifts[nChannels - 1])/tsamp;
    _i_maxshift = GPU_MemoryMap( &_maxshift, sizeof(int) );

    // setup the data buffers
    for( unsigned int i=0; i < maxBuffers; ++i ) {
        _buffersList.append( new DedispersionBuffer(bufferSize) );
    }
    _buffers.reset( &_buffersList );
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
    unsigned int sampleNumber = 0;
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
    GPU_Job* job = new GPU_Job;
    GPU_NVidiaConfiguration config;
    //unsigned int outputSize;
    //GPU_MemoryMap out( &dataOut, outputSize * sizeof(float) );
    //config.addOutputMap( out );
    GPU_MemoryMap in( *buffer, (*buffer)->size() );
    config.addInputMap( GPU_MemoryMap( (*buffer)->getData(), (*buffer)->size() ) );
    config.addInputMap( _i_nsamp );
    config.addInputMap( _i_maxshift );
    config.addInputMap( _i_chans );
    //DedispersionKernel kernel;
    //kernel.setConfiguration( config );
    //job->addKernel( &kernel );
    //job->addCallBack( boost::bind( &DedispersionModule::dataExtract, this, out, dataOut) );
    job->addCallBack( boost::bind( &DedispersionModule::gpuJobFinished, this, job, buffer) );
    submit( job );
}

void DedispersionModule::gpuJobFinished( GPU_Job* job, DedispersionBuffer** buffer ) {
     _buffers.unlock( buffer );
     delete job;
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

} // namespace lofar
} // namespace pelican
