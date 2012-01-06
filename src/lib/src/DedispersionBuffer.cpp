#include "DedispersionBuffer.h"
#include <algorithm>
#include "SpectrumDataSet.h"
#include "WeightedSpectrumDataSet.h"


namespace pelican {

namespace lofar {


/**
 *@details DedispersionBuffer 
 */
DedispersionBuffer::DedispersionBuffer( unsigned int size, unsigned int sampleSize )
   : _sampleSize(sampleSize)
{
    setSampleCapacity(size);
    clear();
}

/**
 *@details
 */
DedispersionBuffer::~DedispersionBuffer()
{
}

void DedispersionBuffer::setSampleCapacity(unsigned int maxSamples)
{
    _nsamp = maxSamples;
    _data.resize( _nsamp * _sampleSize );
}

void DedispersionBuffer::copy( DedispersionBuffer* buf, unsigned int offset )
{
    Q_ASSERT( buf->size() >= size() );
    unsigned s = _data.size() - offset;
    if( s > 0 ) 
        memcpy( &(buf->_data[0]), &_data[offset] , s );
    //buf->_rms = _rms;
    //buf->_mean = _mean;
}

unsigned DedispersionBuffer::spaceRemaining() const {
    return _nsamp - _sampleCount;
}

unsigned DedispersionBuffer::addSamples( WeightedSpectrumDataSet* weightedData, unsigned *sampleNumber ) {

    SpectrumDataSet<float>* streamData = weightedData->dataSet();
    Q_ASSERT( streamData != 0 );
    unsigned int nChannels = streamData->nChannels();
    unsigned int nSubbands = streamData->nSubbands();
    unsigned int nPolarisations = streamData->nPolarisations();
    unsigned int numSamples = streamData->nTimeBlocks();
    if( nSubbands * nChannels * nPolarisations != _sampleSize ) {
        std::cerr  << "DedispersionBuffer: input data sample size(" <<  nSubbands * nChannels * nPolarisations
                   << ") does not match buffer sample size (" << _sampleSize << ")" << std::endl;
        return spaceRemaining();
    }
 
    unsigned maxSamples = std::min( numSamples, spaceRemaining() + *sampleNumber );
    for(unsigned t = *sampleNumber; t < maxSamples; ++t) {
        for (unsigned s = 0; s < nSubbands; ++s) {
            for (unsigned c = 0; c < nChannels; ++c) {
                float* data = streamData->spectrumData(t, s, 0);
                _data[ ((s * nChannels) + c ) * _nsamp  + _sampleCount ] = data[c];
            }
        }
        ++_sampleCount;
    }
    *sampleNumber = maxSamples;
    return spaceRemaining();
}

void DedispersionBuffer::clear() {
    _sampleCount = 0;
}

} // namespace lofar
} // namespace pelican
