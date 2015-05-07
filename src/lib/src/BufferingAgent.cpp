#include "BufferingAgent.h"


namespace pelican {
namespace ampp {

BufferingAgent::BufferingAgent(const DataFetchFunction& fn)
    : QThread()
    , _max_queue_length(3)
    , _halt(false)
    , _fn(fn)
{
    // create some objects to fill
    for(unsigned int i=0; i < _max_queue_length; ++i ) {
        _buffer_objects.push_back(DataBlobHash());
    }
    // assign then to the buffer locking manager
    _buffer.reset(&_buffer_objects);
}

BufferingAgent::~BufferingAgent()
{
    _halt = true;
    if(!_queue.empty()) _buffer.unlock(_queue.front()); // ensure to remove any block 
}

void BufferingAgent::run() {
    _halt = false;
    while(1) {
        if(_halt) return;
        DataBlobHash& hash = *(_buffer.next()); // blocks until ready
        if(_halt) return;
        _fn(hash);
        _queue.push_back(&hash);
    }
}

void BufferingAgent::getData(BufferingAgent::DataBlobHash& hash) {
    // spin until we have data
    do{}
    while(_queue.empty());

    hash.swap(*_queue.front()); // TODO verify this is doing what we think its doing
    _buffer.unlock(_queue.front());
    _queue.pop_front();
}

} // namespace ampp
} // namespace pelican
