#include "DedispersionBuffer.h"
#include <algorithm>
#include "SpectrumDataSet.h"
#include "WeightedSpectrumDataSet.h"


namespace pelican {

namespace lofar {


/**
 *@details DedispersionBuffer 
 */
DedispersionBuffer::DedispersionBuffer( unsigned int size )
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
    Q_ASSERT( buf->size() > size() );
    memcpy( &(buf->_data[0]), &_data[0] + offset * sizeof(float) , (size() - offset) * sizeof(float) );
    //buf->_rms = _rms;
    //buf->_mean = _mean;
}

unsigned DedispersionBuffer::spaceRemaining() const {
    return _nsamp - _sampleCount;
}

unsigned DedispersionBuffer::addSamples( WeightedSpectrumDataSet* weightedData, unsigned *sampleNumber ) {

    SpectrumDataSet<float>* streamData = weightedData->dataSet();
    unsigned int nChannels = streamData->nChannels();
    unsigned int nSubbands = streamData->nSubbands();
    Q_ASSERT( nSubbands * nChannels == _sampleSize );
 
    unsigned maxSamples = std::min( streamData->nTimeBlocks(), spaceRemaining() + *sampleNumber );
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
    return streamData->nTimeBlocks() - maxSamples;
}

void DedispersionBuffer::clear() {
    _sampleCount = 0;
}

} // namespace lofar
} // namespace pelican
