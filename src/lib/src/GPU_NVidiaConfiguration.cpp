#include "GPU_NVidiaConfiguration.h"
#include "GPU_Param.h"


namespace pelican {

namespace lofar {


/**
 *@details GPU_NVidiaConfiguration 
 */
GPU_NVidiaConfiguration::GPU_NVidiaConfiguration()
{
}

/**
 *@details
 */
GPU_NVidiaConfiguration::~GPU_NVidiaConfiguration()
{
    freeResources();
}

void GPU_NVidiaConfiguration::freeResources() {
    _freeMem( _params );
    _params.clear();
    _freeMem( _constantParams.values() );
    _constantParams.clear();
    reset();
}

void GPU_NVidiaConfiguration::reset() {
    _paramIndex = -1;
    _outputs.clear();
}

void GPU_NVidiaConfiguration::syncOutputs() {
    foreach( GPU_Param* param, _outputs ) {
        param->syncDeviceToHost();
    }
}

void* GPU_NVidiaConfiguration::devicePtr( const GPU_MemoryMapOutput& map ) {
     GPU_Param* p = _getParam(map);
     _outputs.insert(p);
     return p->device();
}

void* GPU_NVidiaConfiguration::devicePtr( const GPU_MemoryMapInputOutput& map ) {
     GPU_Param* p = _getParam(map);
     _outputs.insert(p);
     p->syncHostToDevice();
     return p->device();
}

void* GPU_NVidiaConfiguration::devicePtr( const GPU_MemoryMap& map ) {
     GPU_Param* p = _getParam(map);
     p->syncHostToDevice();
     return p->device();
}

//
// Constants are treated differently, as we need to verify that
// its contents are the same as well as merely being a memory
// block of a sufficient size.
//
void* GPU_NVidiaConfiguration::devicePtr( const GPU_MemoryMapConst& map ) {
     if( ! _constantParams.contains(map) ) {
         GPU_Param* p = new GPU_Param( map ) ;
         //if(  cudaPeekAtLastError() ) {
         //    throw( cudaGetErrorString( cudaPeekAtLastError() ) );
        // }
         _constantParams.insert( map, p );
         p->syncHostToDevice(); // consts sync only on creation
     }
     return _constantParams.value(map)->device();
}

//
// This function makes the assumption that the calling sequence
// for resources will be the same in identical configurations.
// Any deviation from that of the original sequence will cause 
// a throw.
//
GPU_Param* GPU_NVidiaConfiguration::_getParam( const GPU_MemoryMap& map ) {
     if( ++_paramIndex >= _params.size() ) {
         // run out of existing params so create a new one
         GPU_Param* p = new GPU_Param( map ) ;
         //if(  cudaPeekAtLastError() ) {
         //    throw( cudaGetErrorString( cudaPeekAtLastError() ) );
         //}
         _params.append( p );
        return p;
     }
     // check that existing parameter matches the request
     // we only check the size
     GPU_Param* p = _params[_paramIndex];
     if( p->size() == map.size() ) {
        p->resetMap(map);
        return p;
     }

     // we cannot satisfy the request
     throw GPUConfigError();
}


void GPU_NVidiaConfiguration::_freeMem( const QList<GPU_Param*>& list ) {
     foreach( GPU_Param* p, list ) {
        delete p;
     }
}

} // namespace lofar
} // namespace pelican
