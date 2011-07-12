#ifndef BLOBSTATISTICS_H
#define BLOBSTATISTICS_H

#include "pelican/data/DataBlob.h"
/**
 * @file BlobStatistics.h
 */

namespace pelican {

namespace lofar {


/**
 * @class BlobStatistics
 *  
 * @brief
 *    simple container class of Statistical measures
 * @details
 * 
 */

class BlobStatistics : public DataBlob
{
    public:
        BlobStatistics( float mean = 0.0f, float rms = 0.0f, float median = 0.0f);
        ~BlobStatistics();
        void reset();
        float rms() const;
        float mean() const;
        float median() const;
        void setRMS(float rms);
        void setMedian(float median);
        void setMean(float mean);

        virtual void serialise(QIODevice& in, QSysInfo::Endian endian);
        virtual void deserialise(QIODevice& in);

    private:
        float _mean, _rms, _median;
};
PELICAN_DECLARE_DATABLOB(BlobStatistics)

} // namespace lofar
} // namespace pelican
#endif // BLOBSTATISTICS_H 
