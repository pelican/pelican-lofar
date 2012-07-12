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
DedispersionBuffer::DedispersionBuffer( unsigned int size, unsigned int sampleSize,
                                        bool invertChannels )
   : _sampleSize(sampleSize), _invertChannels(invertChannels)
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
    unsigned int blobSample;
    // copy the memory
    while( count < samples ) {
        Q_ASSERT( blobIndex > 0 );
        SpectrumDataSetStokes* blob = _inputBlobs[--blobIndex];
        unsigned s = blob->nTimeBlocks();
        sampleNum = samples - count; // remaining samples
        if( sampleNum <= s ) {
            // We have all the samples we need in the current blob
            buf->_sampleCount = 0; // offset position to write to
            blobSample = s - sampleNum;
        } else {
            // Take all the samples from this blob
            buf->_sampleCount = sampleNum - s;// offset position to write to
            blobSample = 0;
        }
        buf->_addSamples( blob, &blobSample, s - blobSample );
        buf->_inputBlobs.push_front( blob );
        count += s;
    }
    buf->_sampleCount = samples;
    return buf->_inputBlobs;
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

    if( _sampleCount == 0 ) {
        // record first sample number
        _firstSample = *sampleNumber;
    }
    unsigned maxSamples = std::min( numSamples, spaceRemaining() + *sampleNumber );
    timerStart(&_addSampleTimer);
    if( _invertChannels ) {
        for(unsigned t = *sampleNumber; t < maxSamples; ++t) {
            for (unsigned s = 0; s < nSubbands; ++s) {
                int bsize = s*nChannels;
                const float* data = streamData->spectrumData(t, nSubbands-1- s, 0);
                for (unsigned c = 0; c < nChannels; ++c) {
                    _timedata[ (bsize + c ) * _nsamp + _sampleCount ] = data[nChannels-1-c];
                }
            }
            ++_sampleCount;
        }
    } else {
        for(unsigned t = *sampleNumber; t < maxSamples; ++t) {
            for (unsigned s = 0; s < nSubbands; ++s) {
                const float* data = streamData->spectrumData(t, s, 0);
                int bsize = s*nChannels;
                for (unsigned c = 0; c < nChannels; ++c) {
                    _timedata[ (bsize + c ) * _nsamp + _sampleCount ] = data[c];
                }
            }
            ++_sampleCount;
        }
    }
    *sampleNumber = maxSamples;
    timerUpdate(&_addSampleTimer);
    timerReport(&_addSampleTimer, "DedispersionBuffer::addSamples");
    return spaceRemaining();
}

void DedispersionBuffer::clear() {
    _sampleCount = 0;
    _inputBlobs.clear();
}

} // namespace lofar
} // namespace pelican
