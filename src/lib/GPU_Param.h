#ifndef GPU_PARAM_H
#define GPU_PARAM_H


/**
 * @file GPU_Param.h
 */

#include "GPU_MemoryMap.h"

namespace pelican {

namespace lofar {
class GPU_MemoryMap;

/**
 * @class GPU_Param
 *  
 * @brief
 *    representation of the GPU_Memorymap on an actual device
 *    i.e. contains the devicePointer
 * @details
 * 
 */

class GPU_Param
{
    public:
        GPU_Param( const GPU_MemoryMap& map );
        virtual ~GPU_Param();
        unsigned long size() const;
        void syncHostToDevice();
        void syncDeviceToHost();
        inline void* operator*() const { return _devicePtr; };
        inline void* device() const { return _devicePtr; };
        /// return the pointer to the host variable
        void* host() const;
        template<typename T> T value() const { return *(static_cast<T*>(host())); }

    private:
        // disable the copy operator
        GPU_Param( const GPU_Param& ) {};

    protected:
        const GPU_MemoryMap _map;
        void* _devicePtr;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_PARAM_H 
