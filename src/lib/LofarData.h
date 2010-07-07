#ifndef LOFARDATA_H
#define LOFARDATA_H


#include "pelican/data/DataBlob.h"

/**
 * @file LofarData.h
 */

namespace pelican {

namespace lofar {
    class LofarStationConfiguration;

/**
 * @class LofarData
 *
 * @ingroup pelican_lofar
 *
 * @brief
 *   A generic Lofar Data container of lofar data and lofar station configuration
 * @details
 * 
 */

class LofarData : public DataBlob
{
    public:
        LofarData( LofarStationConfiguration* config );
        ~LofarData();
        const LofarStationConfiguration& configuration() const;

    private:
        LofarStationConfiguration* _config;
};

} // namespace lofar
} // namespace pelican
#endif // LOFARDATA_H 
