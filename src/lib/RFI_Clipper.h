#ifndef RFI_CLIPPER_H
#define RFI_CLIPPER_H


#include "pelican/modules/AbstractModule.h"

/**
 * @file RFI_Clipper.h
 */

namespace pelican {

namespace lofar {

/**
 * @class RFI_Clipper
 *  
 * @brief
 * 
 * @details
 * 
 */

class RFI_Clipper : public AbstractModule
{
    public:
        RFI_Clipper( const ConfigNode& config );
        ~RFI_Clipper();

    private:
};

} // namespace lofar
} // namespace pelican
#endif // RFI_CLIPPER_H 
