#ifndef LOFARSERVERCLIENT_H
#define LOFARSERVERCLIENT_H


#include "pelican/core/PelicanServerClient.h"

/**
 * @file LofarServerClient.h
 */

namespace pelican {
namespace lofar {

/**
 * @class LofarServerClient
 *
 * @ingroup pelican_lofar
 *
 * @brief
 *    Data Client that connects to a Lofar Enabled Pelican Server
 *
 * @details
 *
 */

class LofarServerClient : public PelicanServerClient
{
    public:
        LofarServerClient(const ConfigNode& configNode,
                const DataTypes& types, const Config* config);
        ~LofarServerClient();

    private:
};

} // namespace lofar
} // namespace pelican

#endif // LOFARSERVERCLIENT_H
