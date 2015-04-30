#ifndef BUFFERINGDATACLIENT_H
#define BUFFERINGDATACLIENT_H


#include "pelican/core/AbstractDataClient.h"
#include "BufferingAgent.h"
#include <deque>

namespace pelican {
namespace lofar {

/**
 * @class BufferingDataClient
 *  
 * @brief
 *     Wraps anouther DataClient in the background, running it in its own thread
 * 
 */

template<class DataClientType>
class BufferingDataClient : public DataClientType
{
    public:
        BufferingDataClient(DataClientType& client);
        ~BufferingDataClient();
        
        // override the getData method only
        virtual DataBlobHash getData(DataBlobHash&);

    private:
        BufferingAgent _agent;
        DataClientType& _client;
};

} // namespace lofar
} // namespace pelican
#include "src/BufferingDataClient.cpp"
#endif // BUFFERINGDATACLIENT_H 
