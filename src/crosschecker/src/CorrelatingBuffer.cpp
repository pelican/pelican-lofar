#include "CorrelatingBuffer.h"


namespace pelican {

namespace lofar {


/**
 *@details CorrelatingBuffer 
 */
CorrelatingBuffer::CorrelatingBuffer( const QString& name, QMap<QString, CorrelatingBuffer*> bufferContainer, 
                                      QObject* parent )
    : QObject(parent), _name(name) 
{
       _correlatedBuffers = bufferContainer; 
        _correlatedBuffers.insert(name,this);

       // TODO initialise _delta
       _delta = 0;
}

/**
 *@details
 */
CorrelatingBuffer::~CorrelatingBuffer()
{
}

void CorrelatingBuffer::add(RTMS_Data& data) 
{
     _buffer.insert(data.startTime(), data );
     QMap<QString, RTMS_Data> corr = findCorrelated( data );
     if( corr.size() > 0  ) {
        corr.insert(_name,data);
        emit foundCorrelation(corr);
        doSomething(corr);
     }
}

QMap<QString, RTMS_Data> CorrelatingBuffer::findCorrelated(const RTMS_Data& d) const
{
     QMap<QString, RTMS_Data> data;
     Timestamp_T startTime = d.startTime();
     Timestamp_T endTime = d.endTime();
     // iterate over all other telescope buffers
     foreach( CorrelatingBuffer* buffer, _correlatedBuffers) {
        if( buffer != this ) {
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
     return data;
}

void CorrelatingBuffer::doSomething(const QMap<QString, RTMS_Data>) const
{
     // TODO
}

} // namespace lofar
} // namespace pelican
