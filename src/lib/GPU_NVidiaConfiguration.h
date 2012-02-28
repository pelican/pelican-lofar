#ifndef GPU_NVIDIACONFIGURATION_H
#define GPU_NVIDIACONFIGURATION_H
#include <QList>
#include <QHash>
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
        void addConstant( const GPU_MemoryMap& param ) {
            _constants.append( param );
        }
        void addOutputMap( const GPU_MemoryMap& param ) { _out.append(param); };
        const QList<GPU_MemoryMap>& outputMaps() const { return _out; };
        const QList<GPU_MemoryMap>& inputMaps() const { return _in; };
        const QList<GPU_MemoryMap>& constants() const { return _constants; };
        const QList<GPU_MemoryMap> allMaps() const { return _in + _constants + _out; };
        void clearInputMaps() { _in.clear(); }
        void clearOutputMaps() { _out.clear(); }

    private:
        QList<GPU_MemoryMap> _out;
        QList<GPU_MemoryMap> _constants;
        QList<GPU_MemoryMap> _in;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_NVIDIACONFIGURATION_H 
