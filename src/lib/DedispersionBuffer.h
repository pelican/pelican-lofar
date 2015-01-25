#ifndef DEDISPERSIONBUFFER_H
#define DEDISPERSIONBUFFER_H
#include <vector>
#include "timer.h"
#include <algorithm>

/**
 * @file DedispersionBuffer.h
 */

namespace pelican {

namespace ampp {
class WeightedSpectrumDataSet;
class SpectrumDataSetStokes;

/**
 * @class DedispersionBuffer
 *  
 * @brief
 *     Data Buffering and mamangement for the Dedispersion Module
 * @details
 * 
 */

class DedispersionBuffer
{
    public:
        DedispersionBuffer( unsigned int size = 0, unsigned int sampleSize = 0,
                            bool invertChannels = false );
        ~DedispersionBuffer();

        /// return the number of samples currently stored in the buffer
        unsigned int numSamples() const { return _sampleCount; };

        size_t size() const { return _timedata.size() * sizeof(float); };
        void setSampleCapacity(unsigned int maxSamples);
        unsigned int numZeros() const { return std::count(_timedata.begin(), _timedata.end(), 0.0); };
        unsigned int elements() const { return _timedata.size(); };
        void fillWithZeros() { _timedata.assign(_timedata.size(), 0.0); };
        /// return the max number of samples that can be fitted in the buffer
        //  set with setSampleCapacity
        unsigned maxSamples() const { return _nsamp; }

        /// return the number of elements in each sample
        unsigned sampleSize() const { return _sampleSize; }

        // remove current data and reset the buffer
        void clear();

        /// import as many samples as possible into the buffer from the 
        // provided data set staring at the sample given by sampleNumber. 
        // The space remaining in the buffer is provided in return value 
        // sampleNumber is updated to the last sample number from
        // the dataset that was included
        unsigned int addSamples( WeightedSpectrumDataSet* weightedData, std::vector<float>& noiseTemplate, unsigned *sampleNumber );

        /// return the amount of empty space in the buffer (in samples)
        inline unsigned int spaceRemaining() const {
            return _nsamp - _sampleCount;
        };
        unsigned int numberOfSamples() const { return _sampleCount; };

        /// return the first sample number (in the first datablob)
        inline unsigned int firstSampleNumber() const { return _firstSample; }

        /// copy data from this object to the supplied DataBuffer object
        //  data is taken from the last offset samples in the 
        //  buffer
        const QList<SpectrumDataSetStokes*>& copy( DedispersionBuffer* buf, std::vector<float>& noiseTemplate, unsigned int samples = 0, unsigned int lastSample = 0);

        /// return the list of input data Blobs used to construct
        //  the data buffer
        inline const QList< SpectrumDataSetStokes* >& inputDataBlobs() const {
                return _inputBlobs; };

        std::vector<float>& getData() { return _timedata; };
        inline float rms() const { return _rms; };
        inline float mean() const { return _mean; };

        void dump( const QString& fileName ) const;
        void dumpbin( const QString& fileName ) const;

    private:
        unsigned int _addSamples( WeightedSpectrumDataSet* data, std::vector<float>& noiseTemplate, unsigned *sampleOffset, unsigned numSamples /* max number fo samples to insert */ );
        unsigned int _addSamples( SpectrumDataSetStokes* data, std::vector<float>& noiseTemplate, unsigned *sampleOffset, unsigned numSamples /* max number fo samples to insert */ );
        QList<SpectrumDataSetStokes* > _inputBlobs;
        std::vector<float> _timedata;
        unsigned int _nsamp;
        unsigned int _sampleCount;
        unsigned int _sampleSize;
        float _mean;
        float _rms;
        bool _invertChannels;
        unsigned int  _firstSample;
        DEFINE_TIMER(_addSampleTimer)
};

} // namespace ampp
} // namespace pelican
#endif // DEDISPERSIONBUFFER_H
