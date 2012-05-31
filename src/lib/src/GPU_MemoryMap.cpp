#include "GPU_MemoryMap.h"
#include <QHash>
#include <QPair>


namespace pelican {

namespace lofar {


/**
 *@details GPU_MemoryMap 
 */
GPU_MemoryMap::GPU_MemoryMap( void* host_address, unsigned long s )
{
    _set(host_address, s);
}

/**
 *@details
 */
GPU_MemoryMap::~GPU_MemoryMap()
{
    runCallBacks();
}

void GPU_MemoryMap::_set(void* host_address, unsigned long s) {
     _host = host_address;
     _size = s;
     _hash = ::qHash( QPair<void*,unsigned long>(_host, _size ) );
}

bool GPU_MemoryMap::operator==(const GPU_MemoryMap& m) const
{
     return (m._host == _host) && ( m._size == _size );
}

void GPU_MemoryMap::runCallBacks() const {
    foreach( const GPU_MemoryMap::CallBackT& fn, _callbacks ) {
       fn();
    }
    _callbacks.clear(); // must clear after making the calls
                        // to avoid multiple calls to the
                        // same function
}

/// Compute a hash value for use with QHash (uses the hash member function).
uint qHash(const GPU_MemoryMap& map) {
    return map.qHash();
}

} // namespace lofar
} // namespace pelican
