#include <QFile>
#include <QTextStream>
#include "DedispersionBuffer.h"
#include <algorithm>
#include "SpectrumDataSet.h"


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
    _timedata.resize( maxSamples * _sampleSize );
}

void DedispersionBuffer::dump( const QString& fileName ) const {
    QFile file(fileName);
    if (QFile::exists(fileName)) QFile::remove(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return;
    QTextStream out(&file);

    for (int c = 0; c < _timedata.size(); ++c) {
        out << QString::number(_timedata[c], 'g' ) << QString(((c+1)%_nsamp == 0)?"\n":" ");
    }
    file.close();
}

const QList<SpectrumDataSetStokes*>& DedispersionBuffer::copy( DedispersionBuffer* buf, unsigned int samples )
{
    unsigned int count = 0;
    unsigned int blobIndex = _inputBlobs.size();
    unsigned int sampleNum;
    unsigned int blobSample = 0;
    while( count < samples ) {
        Q_ASSERT( blobIndex > 0 );
        SpectrumDataSetStokes* blob = _inputBlobs[--blobIndex];
        unsigned s = blob->nTimeBlocks();
        sampleNum = samples - count; // remaining samples
        if( sampleNum <= s ) {
            buf->_sampleCount = 0; // offset position to write to
            blobSample = s - sampleNum;
        } else {
            buf->_sampleCount = sampleNum - s;// offset position to write to
        }
        buf->_addSamples( blob, &blobSample, s - blobSample );
        buf->_inputBlobs.push_front( blob );
        count += s;
    }
    buf->_sampleCount = samples;
    return buf->_inputBlobs;
}

unsigned DedispersionBuffer::spaceRemaining() const {
    return _nsamp - _sampleCount;
}

unsigned DedispersionBuffer::addSamples( SpectrumDataSetStokes* streamData, unsigned *sampleNumber ) {
    if( ! _inputBlobs.contains(streamData) )
        _inputBlobs.append(streamData);
    unsigned int numSamples = streamData->nTimeBlocks();
    return _addSamples( streamData, sampleNumber, numSamples );
}

unsigned DedispersionBuffer::_addSamples( SpectrumDataSetStokes* streamData, 
                                          unsigned *sampleNumber, unsigned numSamples ) {
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
            const float* data = streamData->spectrumData(t, s, 0);
            for (unsigned c = 0; c < nChannels; ++c) {
                // rearrange to have time samples in inner loop
                _timedata[ ((s * nChannels) + c ) * _nsamp + _sampleCount ] = data[c];
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
