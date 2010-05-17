#ifndef LOFARSTREAMDATACLIENT_H
#define LOFARSTREAMDATACLIENT_H

#include "pelican/core/DirectStreamDataClient.h"

/**
 * @file LofarStreamDataClient.h
 */

namespace pelican {
namespace lofar {

/**
 * @class LofarStreamDataClient
 *
 * @brief
 *    Lofar Data Client to Connect directly to the LOFAR station
 *    output stream.
 *
 * @details
 *
 */

class LofarStreamDataClient : public DirectStreamDataClient
{
    public:
        LofarStreamDataClient(const ConfigNode& configNode,
                const DataTypes& types, const Config* config);
        ~LofarStreamDataClient();

    private:
};

} // namespace lofar
} // namespace pelican

#endif // LOFARSTREAMDATACLIENT_H
