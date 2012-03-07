#include <QFile>
#include <QTextStream>
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

void DedispersionBuffer::dump( const QString& fileName ) const {
    QFile file(fileName);
    if (QFile::exists(fileName)) QFile::remove(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return;
    QTextStream out(&file);
     
    for (int c = 0; c < _data.size(); ++c) {
        out << QString::number(_data[c], 'g' ) << QString(((c+1)%_nsamp == 0)?"\n":" ");
    }
    file.close();
}

void DedispersionBuffer::copy( DedispersionBuffer* buf, unsigned int samples )
{
    unsigned int count = 0;
    unsigned int blobIndex = _inputBlobs.size();
    unsigned int sampleNum;
    unsigned int blobSample = 0;
    while( count < samples ) {
        Q_ASSERT( blobIndex > 0 );
        WeightedSpectrumDataSet* blob = _inputBlobs[--blobIndex];
        unsigned s = blob->dataSet()->nTimeBlocks();
        sampleNum = samples - count; // remaining samples
        if( sampleNum <= s ) {
            buf->_sampleCount = 0;// ofset position to write to
            blobSample = s - sampleNum;
        } else {
            buf->_sampleCount = sampleNum - s;// ofset position to write to
        }
        buf->_addSamples( blob, &blobSample, s - blobSample );
        buf->_inputBlobs.push_front( blob );
        count += s;
    } 
    buf->_sampleCount = samples;
    //Q_ASSERT( count == samples );
}

unsigned DedispersionBuffer::spaceRemaining() const {
    return _nsamp - _sampleCount;
}

unsigned DedispersionBuffer::addSamples( WeightedSpectrumDataSet* weightedData, unsigned *sampleNumber ) {
    _inputBlobs.append(weightedData);
    unsigned int numSamples = weightedData->dataSet()->nTimeBlocks();
    return _addSamples( weightedData, sampleNumber, numSamples );
}

unsigned DedispersionBuffer::_addSamples( WeightedSpectrumDataSet* weightedData, 
                                          unsigned *sampleNumber, unsigned numSamples ) {
    SpectrumDataSet<float>* streamData = weightedData->dataSet();
    Q_ASSERT( streamData != 0 );
    unsigned int nChannels = streamData->nChannels();
    unsigned int nSubbands = streamData->nSubbands();
    unsigned int nPolarisations = streamData->nPolarisations();
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
                // rearrange to have time samples in inner loop
               _data[ ((s * nChannels) + c ) * _nsamp + _sampleCount ] = data[c];
            }
        }
        ++_sampleCount;
    }
    *sampleNumber = maxSamples;
    return spaceRemaining();
}

void DedispersionBuffer::clear() {
    _sampleCount = 0;
    _inputBlobs.clear();
}

} // namespace lofar
} // namespace pelican
