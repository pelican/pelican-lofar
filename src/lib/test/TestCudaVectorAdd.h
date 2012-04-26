#ifndef TESTCUDAVECTORADD_H
#define TESTCUDAVECTORADD_H

#include <QList>
#include "GPU_Kernel.h"
#include "GPU_MemoryMap.h"

/**
 * @file TestCudaVectorAdd.h
 */

namespace pelican {

namespace lofar {
class GPU_Param;

/**
 * @class TestCudaVectorAdd
 *  
 * @brief
 *    A Vector Add kernle for testing NVidia job submissions
 * @details
 * 
 */

class TestCudaVectorAdd : public GPU_Kernel
{
    public:
        TestCudaVectorAdd();
        virtual ~TestCudaVectorAdd();
        virtual void run( GPU_NVidia& );
        inline void addConstant( const GPU_MemoryMap& map ) { _vec1 = map; };
        inline void addInputMap( const GPU_MemoryMap& map ) { _vec2 = map; };
        inline void addOutputMap( const GPU_MemoryMap& map ) { _vecOut = map; };

    private:
        GPU_MemoryMapConst _vec1;
        GPU_MemoryMap _vec2;
        GPU_MemoryMapOutput _vecOut;
};

} // namespace lofar
} // namespace pelican
#endif // TESTCUDAVECTORADD_H 
