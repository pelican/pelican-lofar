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
        GPU_MemoryMap(  );
        virtual ~GPU_MemoryMap();
        inline char* start() { return _start; };
        inline char* destination() { return _destination; };
        inline unsigned long size() { return _size; }

    private:
        char* _start;
        char* _destination;
        unsigned long _size;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_MEMORYMAP_H 
