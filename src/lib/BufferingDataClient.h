#ifndef BUFFERINGDATACLIENT_H
#define BUFFERINGDATACLIENT_H


#include "pelican/core/AbstractDataClient.h"
#include "BufferingAgent.h"

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
        BufferingDataClient(const ConfigNode& configNode, const DataTypes& types, const Config* config);
        ~BufferingDataClient();
        
        // overriden methods here
        virtual pelican::AbstractDataClient::DataBlobHash getData(pelican::AbstractDataClient::DataBlobHash&);

    private:
        // fn to execute the event loop
        void exec();

    private:
        BufferingAgent _agent;
	bool _halt;
};

} // namespace ampp
} // namespace pelican
#include "src/BufferingDataClient.cpp"
#endif // BUFFERINGDATACLIENT_H 
