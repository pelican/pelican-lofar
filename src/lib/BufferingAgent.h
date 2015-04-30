#ifndef BUFFERINGAGENT_H
#define BUFFERINGAGENT_H


#include "pelican/core/AbstractDataClient.h"
#include "QThread.h"

/**
 * @file BufferingAgent.h
 */

namespace pelican {

namespace lofar {

/**
 * @class BufferingAgent
 *  
 * @brief
 *    A dedicated thread for running 
 * @details
 * 
 */

class BufferingAgent : public QThread
{
    public:
        BufferingAgent(pelican::AbstractDataClient&);
        ~BufferingAgent();

        void run();

    private:
        unsinged int _max_queue_length;
        bool _halt;
        pelican::AbstractDataClient& _client;
        std::deque<pelican::AbstractDataClient::DataBlobHash> _queue;
};

} // namespace lofar
} // namespace pelican
#endif // BUFFERINGAGENT_H 
