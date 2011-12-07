#ifndef GPU_NVIDIACONFIGURATION_H
#define GPU_NVIDIACONFIGURATION_H
#include <QList>
#include "GPU_MemoryMap.h"


/**
 * @file GPU_NVidiaConfiguration.h
 */

namespace pelican {

namespace lofar {

/**
 * @class GPU_NVidiaConfiguration
 *  
 * @brief
 *    Describes the memory requirements for an NVidia card
 * @details
 * 
 */

class GPU_NVidiaConfiguration
{
    public:
        GPU_NVidiaConfiguration(  );
        ~GPU_NVidiaConfiguration();
        void addInputMap( const GPU_MemoryMap& param ) { _in.append(param); };
        void addOutputMap( const GPU_MemoryMap& param ) { _out.append(param); };
        const QList<GPU_MemoryMap> outputMaps() const { return _out; };
        const QList<GPU_MemoryMap> inputMaps() const { return _in; };
        const QList<GPU_MemoryMap> allMaps() const { return _in + _out; };

    private:
        QList<GPU_MemoryMap> _out;
        QList<GPU_MemoryMap> _in;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_NVIDIACONFIGURATION_H 
