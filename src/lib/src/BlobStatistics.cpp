#include "BlobStatistics.h"
#include <QDataStream>

namespace pelican {

namespace lofar {


/**
 *@details BlobStatistics 
 */

BlobStatistics::BlobStatistics(float mean, float rms, float median )
   : DataBlob("BlobStatistics"), _mean(mean), _rms(rms), _median(median)
{
}
/**
 *@details
 */
BlobStatistics::~BlobStatistics()
{
}

void BlobStatistics::setRMS(float rms)
{
    _rms = rms;
}

void BlobStatistics::setMedian(float median)
{
    _median = median;
}

void BlobStatistics::setMean(float mean)
{
    _mean = mean;
}

float BlobStatistics::median() const
{
    return _median;
}

float BlobStatistics::rms() const
{
    return _rms;
}

float BlobStatistics::mean() const
{
    return _mean;
}

void BlobStatistics::reset()
{
    _mean = 0.0f;
    _median = 0.0f;
    _rms = 0.0f;
}

void BlobStatistics::serialise(QIODevice& device, QSysInfo::Endian )
{
    QDataStream out(&device);
    out.setVersion(QDataStream::Qt_4_0);
    out << _mean;
    out << _rms;
    out << _median;
}

void BlobStatistics::deserialise(QIODevice& device)
{
    QDataStream in(&device);
    in.setVersion(QDataStream::Qt_4_0);
    in >> _mean;
    in >> _rms;
    in >> _median;
}

} // namespace lofar
} // namespace pelican
