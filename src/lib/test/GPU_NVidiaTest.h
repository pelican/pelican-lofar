#ifndef GPU_NVIDIATEST_H
#define GPU_NVIDIATEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file GPU_NVidiaTest.h
 */

namespace pelican {

namespace lofar {

/**
 * @class GPU_NVidiaTest
 *  
 * @brief
 *    unit test for the GPU_NVidia plugin
 * @details
 * 
 */

class GPU_NVidiaTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( GPU_NVidiaTest );
        CPPUNIT_TEST( test_managedCard );
        CPPUNIT_TEST( test_multipleJobs );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_managedCard();
        void test_multipleJobs();

    public:
        GPU_NVidiaTest(  );
        ~GPU_NVidiaTest();

    private:
};

} // namespace lofar
} // namespace pelican
#endif // GPU_NVIDIATEST_H 
