#ifndef GPU_MEMORYMAPTEST_H
#define GPU_MEMORYMAPTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file GPU_MemoryMapTest.h
 */

namespace pelican {

namespace ampp {

/**
 * @class GPU_MemoryMapTest
 *  
 * @brief
 * 
 * @details
 * 
 */

class GPU_MemoryMapTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( GPU_MemoryMapTest );
        CPPUNIT_TEST( test_method );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();

    public:
        GPU_MemoryMapTest(  );
        ~GPU_MemoryMapTest();

    private:
};

} // namespace ampp
} // namespace pelican
#endif // GPU_MEMORYMAPTEST_H 
