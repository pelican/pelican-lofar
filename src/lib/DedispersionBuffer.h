#ifndef DEDISPERSIONBUFFER_H
#define DEDISPERSIONBUFFER_H
#include <QVector>


/**
 * @file DedispersionBuffer.h
 */

namespace pelican {

namespace lofar {
class WeightedSpectrumDataSet;

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
        DedispersionBuffer( unsigned int size = 0 );
        ~DedispersionBuffer();

        // return the size of the buffer
        unsigned int numSamples() const { return _sampleCount; };
        size_t size() const { return _data.size() * sizeof(float); };
        void setSampleCapacity(unsigned int maxSamples);
        
        // remove current data and reset the buffer
        void clear();

        /// import as many samples as poosibkle into the buffer from the 
        // provided data set. The return value represents the remaining samples
        // for which there was no space.
        // the sampleNumber is updated to be the last sample number that was included
        unsigned int addSamples( WeightedSpectrumDataSet* weightedData, unsigned *sampleNumber );

        /// return the amount of empty space in the buffer
        unsigned int spaceRemaining() const;

        /// copy data from this object to the supplied DataBuffer object
        //  data is taken from the offset position to the end and
        //  inserted at the beginning of the provided object
        void copy( DedispersionBuffer* buf, unsigned int offset = 0 );

        float* getData() { return &_data[0]; };
        inline float rms() const { return _rms; };
        inline float mean() const { return _mean; };

    private:
        QVector<float> _data;
        unsigned int _nsamp;
        unsigned int _sampleCount;
        unsigned int _sampleSize;
        float _mean;
        float _rms;
};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONBUFFER_H 
