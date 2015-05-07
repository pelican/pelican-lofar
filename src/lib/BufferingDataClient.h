#ifndef BUFFERINGDATACLIENT_H
#define BUFFERINGDATACLIENT_H


#include "pelican/core/AbstractDataClient.h"
#include "BufferingAgent.h"
#include <deque>

namespace pelican {
namespace ampp {

/**
 * @class BufferingDataClient
 *  
 * @brief
 *     Wraps another DataClient in the background, running it in its own thread
 * 
 */

template<class DataClientType>
class BufferingDataClient : public DataClientType
{
    public:
        BufferingDataClient(DataClientType& client);
        ~BufferingDataClient();
        
        // override the getData method only
        virtual BufferingAgent::DataBlobHash getData(BufferingAgent::DataBlobHash&);

    private:
        BufferingAgent _agent;
};

} // namespace ampp
} // namespace pelican
#include "src/BufferingDataClient.cpp"
#endif // BUFFERINGDATACLIENT_H 
