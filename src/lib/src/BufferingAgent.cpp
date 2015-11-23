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

void BufferingAgent::stop()
{
    _halt = true;
    quit();
}

void BufferingAgent::getData(BufferingAgent::DataBlobHash& hash) {
    // spin until we have data
    do{} // TODO verify empty() is thread safe (and _queue.pop_front() call later on particularly when the _buffer is still filling slots)
    while(_queue.empty());

    DataBlobHash* tmp = _queue.front();
    hash.swap(*tmp); // TODO verify this is doing what we think its doing
    _queue.pop_front(); // remove from queue before unblocking the other thread to reduce conflicts
    _buffer.unlock(tmp);
}

} // namespace ampp
} // namespace pelican
