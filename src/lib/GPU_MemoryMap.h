#ifndef GPU_MEMORYMAP_H
#define GPU_MEMORYMAP_H


/**
 * @file GPU_MemoryMap.h
 */

namespace pelican {

namespace lofar {

/**
 * @class GPU_MemoryMap
 *  
 * @brief
 *     A base class to transfer mark out data 
 *     to be transfered between different devices
 * @details
 * 
 */

class GPU_MemoryMap
{
    public:
        GPU_MemoryMap( void* host_address, unsigned long bytes );
        virtual ~GPU_MemoryMap();
        inline void* hostPtr() { return _host; };
        inline unsigned long size() { return _size; }
        bool operator==(const GPU_MemoryMap&);

    private:
        void* _host;
        unsigned long _size;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_MEMORYMAP_H 
