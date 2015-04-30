#include "BufferingAgent.h"


namespace pelican {
namespace lofar {

BufferingAgent::BufferingAgent(AbstractDataClient& client)
    : QThread()
    , _max_queue_length(3)
    , _halt(false)
    , _client(client)
{
}

BufferingAgent::~BufferingAgent()
{
    _halt = false;
}

void BufferingAgent::run() {
    _halt = false;
    while(1) {
        while(_queue.size < _max_queue_length) {
            if(_halt) return;
            DataBlobHash hash; // TODO check how this is done in the history mechanism and copy that (recycling)
            DataClientType::getData(hash);
            _queue.push_back(hash);
        } 
    }
}

void BufferingAgent::getData(DataBlobHash& hash) {
    // spin until we have data
    do{}
    while(_queue.empty());

    hash =  _queue.front();
    _queue.pop_front();
}

} // namespace lofar
} // namespace pelican
