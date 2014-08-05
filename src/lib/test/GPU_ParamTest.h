#ifndef GPU_PARAMTEST_H
#define GPU_PARAMTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file GPU_ParamTest.h
 */

namespace pelican {

namespace ampp {

/**
 * @class GPU_ParamTest
 *  
 * @brief
 *     Unit Test for the GPU_Param class
 * @details
 * 
 */

class GPU_ParamTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( GPU_ParamTest );
        CPPUNIT_TEST( test_memoryLeak );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_memoryLeak();

    public:
        GPU_ParamTest(  );
        ~GPU_ParamTest();

    private:
};

} // namespace ampp
} // namespace pelican
#endif // GPU_PARAMTEST_H 
