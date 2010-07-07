#ifndef LOFARSTATIONCONFIGURATION_H
#define LOFARSTATIONCONFIGURATION_H


#include "pelican/data/DataBlob.h"

/**
 * @file LofarStationConfiguration.h
 */

namespace pelican {

namespace lofar {

/**
 * @class LofarStationConfiguration
 *
 * @ingroup pelican_lofar
 *
 * @brief
 *   Contains inforamtion about the Lofar Station setup
 * @details
 * 
 */

class LofarStationConfiguration : public DataBlob
{
    public:
        LofarStationConfiguration(  );
        ~LofarStationConfiguration();

    private:
};

} // namespace lofar
} // namespace pelican
#endif // LOFARSTATIONCONFIGURATION_H 
