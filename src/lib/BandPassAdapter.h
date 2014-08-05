#ifndef BANDPASSADAPTER_H
#define BANDPASSADAPTER_H


#include "pelican/core/AbstractServiceAdapter.h"


/**
 * @file BandPassAdapter.h
 */

namespace pelican {

namespace ampp {

/**
 * @class BandPassAdapter
 *  
 * @brief Fills a BandPass object from a suitable bandpass data stream.
 *
 * @details
 */

class BandPassAdapter : public AbstractServiceAdapter
{
    public:
	/// Constructor
        BandPassAdapter( const ConfigNode&  );
	/// Destructor
        ~BandPassAdapter();

        /// Method to deserialise chunks of memory provided by the IO device.
        void deserialise(QIODevice* device);

    private:
};

PELICAN_DECLARE_ADAPTER(BandPassAdapter)

} // namespace ampp
} // namespace pelican
#endif // BANDPASSADAPTER_H 
