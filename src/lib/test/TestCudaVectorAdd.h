#ifndef TESTCUDAVECTORADD_H
#define TESTCUDAVECTORADD_H


#include "GPU_Kernel.h"

/**
 * @file TestCudaVectorAdd.h
 */

namespace pelican {

namespace lofar {

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
        virtual void run( const std::vector<void*>& devicePointers );

    private:
};

} // namespace lofar
} // namespace pelican
#endif // TESTCUDAVECTORADD_H 
