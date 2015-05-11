#ifndef BUFFERINGAGENT_H
#define BUFFERINGAGENT_H


#include "pelican/core/AbstractDataClient.h"
#include "LockingContainer.hpp"
#include <QList>
#include <QThread>
#include <boost/function.hpp>
#include <deque>

/**
 * @file BufferingAgent.h
 */

namespace pelican {

namespace ampp {

/**
 * @class BufferingAgent
 *  
 * @brief
 *    A dedicated thread for buffering data 
 * @details
 * 
 */

class BufferingAgent : public QThread
{
    public:
        typedef pelican::AbstractDataClient::DataBlobHash DataBlobHash;
        typedef boost::function1<void, DataBlobHash&> DataFetchFunction;

    public:
        BufferingAgent( const DataFetchFunction& fn );
        ~BufferingAgent();

        void run();
        void stop();

        void getData(DataBlobHash& hash);

    private:
        unsigned int _max_queue_length;
        bool _halt;
        DataFetchFunction _fn;
        std::deque<DataBlobHash*> _queue; // objects ready for serving
        QList<DataBlobHash> _buffer_objects;
        LockingContainer<DataBlobHash> _buffer;
};

} // namespace ampp
} // namespace pelican
#endif // BUFFERINGAGENT_H 
