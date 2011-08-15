#include "CorrelatedBufferManager.h"
#include "CorrelatingBuffer.h"


namespace pelican {

namespace lofar {


/**
 *@details CorrelatedBufferManager 
 */
CorrelatedBufferManager::CorrelatedBufferManager( QObject* parent )
    : QObject(parent)
{
    _delta = 0;
}

/**
 *@details
 */
CorrelatedBufferManager::~CorrelatedBufferManager()
{
}

CorrelatingBuffer* CorrelatedBufferManager::newBuffer(const QString& name)
{
    CorrelatingBuffer* buf = new CorrelatingBuffer(this);
    _buffers.insert(name, buf);
    connect(buf, SIGNAL(dataAdded(const RTMS_Data&)), 
                 SLOT( dataReceived(const RTMS_Data&) ) );
    return buf;
}

void CorrelatedBufferManager::dataReceived(const RTMS_Data& data)
{
    _findCorrelated(data, static_cast<CorrelatingBuffer*>(sender()) );
}

QMap<QString, RTMS_Data> CorrelatedBufferManager::_findCorrelated(
        const RTMS_Data& d, const CorrelatingBuffer* exclude)
{
     QMap<QString, RTMS_Data> data;
     CorrelatingBuffer::Timestamp_T startTime = d.startTime();
     CorrelatingBuffer::Timestamp_T endTime = d.endTime();
     // iterate over all other telescope buffers
     foreach( const CorrelatingBuffer* buffer, _buffers ) {
        if( buffer != exclude ) {
            foreach( const RTMS_Data& bd, buffer->_buffer ) {
                // check times of the two RTMS data to see if
                // they are correlated
                // TODO improve me
                if( ( bd.startTime() - _delta < startTime 
                        && startTime < bd.endTime() + _delta)
                    || ( startTime - _delta < bd.startTime() 
                        && bd.startTime() < endTime + _delta)
                  )
                {
                     // we have a correlation! Add it to our return data
                     data.insert(buffer->_name,bd);
                }
            }
        }
     }
     if( data.size() > 0 ) { emit foundCorrelation(data); }
     return data;
}

} // namespace lofar
} // namespace pelican
