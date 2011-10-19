#ifndef GPU_NVIDIA_H
#define GPU_NVIDIA_H


#include "GPU_Resource.h"

/**
 * @file GPU_NVidia.h
 */

namespace pelican {

namespace lofar {

/**
 * @class GPU_NVidia
 *  
 * @brief
 * 
 * @details
 * 
 */

class GPU_NVidia : public GPU_Resource
{
    public:
        GPU_NVidia(  );
        ~GPU_NVidia();
        virtual void run( const GPU_Job& job );

    private:
};

} // namespace lofar
} // namespace pelican
#endif // GPU_NVIDIA_H 
