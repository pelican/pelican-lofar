#ifndef LOFARSTREAMDATACLIENT_H
#define LOFARSTREAMDATACLIENT_H

#include "pelican/core/DirectStreamDataClient.h"

/**
 * @file LofarStreamDataClient.h
 */

using namespace pelican;

namespace pelicanLofar {

/**
 * @class LofarStreamDataClient
 *
 * @brief
 *    Lofar Data Client to Connecdt directly to the LOFAR station
 *    output stream
 * @details
 *
 */

class LofarStreamDataClient : public DirectStreamDataClient
{
    public:
        LofarStreamDataClient( ConfigNode&, const pelican::DataTypes& );
        ~LofarStreamDataClient();

    private:
};

} // namespace pelicanLofar
#endif // LOFARSTREAMDATACLIENT_H
