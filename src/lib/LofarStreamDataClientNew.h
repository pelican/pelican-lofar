#ifndef LOFARSTREAMDATACLIENT_NEW_H
#define LOFARSTREAMDATACLIENT_NEW_H

#include "pelican/core/DirectStreamDataClient.h"

/**
 * @file LofarStreamDataClientNew.h
 */

namespace pelican {
namespace lofar {

/**
 * @class LofarStreamDataClientNew
 *
 * @ingroup pelican_lofar
 *
 * @brief
 *    Lofar Data Client to Connect directly to the LOFAR station
 *    output stream.
 *
 * @details
 *
 */

class LofarStreamDataClientNew : public DirectStreamDataClient
{
    public:
        LofarStreamDataClientNew(const ConfigNode& configNode,
                const DataTypes& types, const Config* config);
        ~LofarStreamDataClientNew();

    private:
};

PELICAN_DECLARE_CLIENT(LofarStreamDataClientNew)

} // namespace lofar
} // namespace pelican

#endif // LOFARSTREAMDATACLIENT_H
