#ifndef BUFFERINGDATACLIENT_H
#define BUFFERINGDATACLIENT_H


#include "pelican/core/AbstractDataClient.h"
#include "BufferingAgent.h"

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
        BufferingDataClient(const ConfigNode& configNode, const DataTypes& types, const Config* config);
        ~BufferingDataClient();
        
        // overriden methods here
        virtual DataBlobHash getData(pelican::AbstractDataClient::DataBlobHash&);

    private:
        BufferingAgent _agent;
};

} // namespace lofar
} // namespace pelican
#include "src/BufferingDataClient.cpp"
#endif // BUFFERINGDATACLIENT_H 
