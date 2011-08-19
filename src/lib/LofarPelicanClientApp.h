#ifndef LOFARPELICANCLIENTAPP_H
#define LOFARPELICANCLIENTAPP_H

#include "pelican/utility/Config.h"
#include <QMap>

/**
 * @file LofarPelicanClientApp.h
 */

namespace pelican {
class AbstractBlobClient;
class ThreadedDataBlobClient;

namespace lofar {

/**
 * @class LofarPelicanClientApp
 *  
 * @brief
 *     convenience class for applications wishing to do downstream
 * processing of data streamed from a pelican-lofar pipeline
 * 
 * @details
 *  
 */

class LofarPelicanClientApp
{
    public:
        typedef QMap<QString,ThreadedDataBlobClient*> ClientMapContainer_T;

    public:
        LofarPelicanClientApp(int argc, char** argv, const Config::TreeAddress& baseNode );
        ~LofarPelicanClientApp();
        ClientMapContainer_T clients() const;
        ConfigNode config(const Config::TreeAddress address) const;

    protected:
        pelican::Config _config;
        Config::TreeAddress _address;

    private:
        ClientMapContainer_T _clients;
};

} // namespace lofar
} // namespace pelican
#endif // LOFARPELICANCLIENTAPP_H 
