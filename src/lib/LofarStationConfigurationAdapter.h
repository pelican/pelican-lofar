#ifndef LOFARSTATIONCONFIGURATIONADAPTER_H
#define LOFARSTATIONCONFIGURATIONADAPTER_H


#include "pelican/core/AbstractServiceAdapter.h"

/**
 * @file LofarStationConfigurationAdapter.h
 */

namespace pelican {

    class ConfigNode;

namespace lofar {

/**
 * @class LofarStationConfigurationAdapter
 *  
 * @brief
 *    translates the station configuration file into a LofarStationConfiguration object
 * @details
 * 
 */

class LofarStationConfigurationAdapter : public AbstractServiceAdapter
{
    public:
        LofarStationConfigurationAdapter( const ConfigNode& config  );
        ~LofarStationConfigurationAdapter();
        void deserialise(QIODevice&);

    private:
};

} // namespace lofar
} // namespace pelican
#endif // LOFARSTATIONCONFIGURATIONADAPTER_H 
