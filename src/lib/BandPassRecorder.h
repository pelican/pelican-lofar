#ifndef BANDPASSRECORDER_H
#define BANDPASSRECORDER_H


#include "AbstractModule.h"

/**
 * @file BandPassRecorder.h
 */

namespace pelican {

namespace lofar {

/**
 * @class BandPassRecorder
 *  
 * @brief
 *    Measure a suitable BandPass from an incoming data stream
 * @details
 * 
 */

class BandPassRecorder : public AbstractModule
{
    public:
        BandPassRecorder( const ConfigNode& config );
        ~BandPassRecorder();
        void run( BandPass* bp );

    private:
};

PELICAN_DECLARE_MODULE(BandPassRecorder)

} // namespace lofar
} // namespace pelican
#endif // BANDPASSRECORDER_H 
