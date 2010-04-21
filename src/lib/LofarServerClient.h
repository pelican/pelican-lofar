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
 * @brief
 *    Data Client that connects to a Lofar Enabled Pelican Server
 *
 * @details
 *
 */

class LofarServerClient : public PelicanServerClient
{
    public:
        LofarServerClient(const ConfigNode& config);
        ~LofarServerClient();

    private:
};

} // namespace lofar
} // namespace pelican

#endif // LOFARSERVERCLIENT_H
