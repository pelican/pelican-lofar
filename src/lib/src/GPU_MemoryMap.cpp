#include "GPU_MemoryMap.h"


namespace pelican {

namespace lofar {


/**
 *@details GPU_MemoryMap 
 */
GPU_MemoryMap::GPU_MemoryMap( void* host_address, unsigned long s )
    : _host(host_address), _size(s)
{
}

/**
 *@details
 */
GPU_MemoryMap::~GPU_MemoryMap()
{
}

bool GPU_MemoryMap::operator==(const GPU_MemoryMap& m)
{
     return (m._host == _host) && ( m._size == _size );
}

} // namespace lofar
} // namespace pelican
