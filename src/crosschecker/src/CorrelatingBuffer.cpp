#include "CorrelatingBuffer.h"
#include "pelican/output/Stream.h"


namespace pelican {

namespace ampp {


/**
 *@details CorrelatingBuffer 
 *    this is a proviate contructor, use newBuffer()
 */
CorrelatingBuffer::CorrelatingBuffer( CorrelatedBufferManager* manager, QObject* parent )
    : QObject(parent), _manager(manager)
{
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
     emit dataAdded(data);
/*
     QMap<QString, RTMS_Data> corr = _manager->findCorrelated( data );
     if( corr.size() > 0  ) {
        corr.insert(_name,data);
        doSomething(corr);
     }
*/
}

void CorrelatingBuffer::doSomething(const QMap<QString, RTMS_Data>) const
{
     // TODO
}

void CorrelatingBuffer::newData(const Stream& stream)
{
    Q_ASSERT( stream.type() == "RTMS_Data" );
    add(*(static_cast<RTMS_Data*>(stream.data().get())));
}
} // namespace ampp
} // namespace pelican
