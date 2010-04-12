#ifndef LOFARSTREAMDATACLIENT_H
#define LOFARSTREAMDATACLIENT_H


#include "pelican/DirectStreamDataClient.h"

/**
 * @file LofarStreamDataClient.h
 */

using namespace pelican;

namespace pelicanLofar {

/**
 * @class LofarStreamDataClient
 *  
 * @brief
 * 
 * @details
 * 
 */

class LofarStreamDataClient : public DirectStreamDataClient
{
    public:
        LofarStreamDataClient(  );
        ~LofarStreamDataClient();

    private:
};

} // namespace pelicanLofar
#endif // LOFARSTREAMDATACLIENT_H 
