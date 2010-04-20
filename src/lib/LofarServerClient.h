#ifndef LOFARSERVERCLIENT_H
#define LOFARSERVERCLIENT_H


#include "pelican/core/PelicanServerClient.h"

/**
 * @file LofarServerClient.h
 */

using namespace pelican;

namespace pelicanLofar {

/**
 * @class LofarServerClient
 *  
 * @brief
 *    Data Client that connects to a Lofar Enabled Pelican Server
 * @details
 * 
 */

class LofarServerClient : public PelicanServerClient
{
    public:
        LofarServerClient( const ConfigNode& config, const DataTypes& types );
        ~LofarServerClient();

    private:
};

} // namespace pelicanLofar
#endif // LOFARSERVERCLIENT_H 
