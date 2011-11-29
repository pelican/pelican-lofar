#ifndef GPU_MANAGERTEST_H
#define GPU_MANAGERTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file GPU_ManagerTest.h
 */

namespace pelican {

namespace lofar {

/**
 * @class GPU_ManagerTest
 *  
 * @brief
 *    Unit test for the GPU_Manager
 * @details
 * 
 */

class GPU_ManagerTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( GPU_ManagerTest );
        CPPUNIT_TEST( test_submit );
        CPPUNIT_TEST( test_submitMultiCards );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_submit();
        void test_submitMultiCards();

    public:
        GPU_ManagerTest(  );
        ~GPU_ManagerTest();

    private:
};

} // namespace lofar
} // namespace pelican
#endif // GPU_MANAGERTEST_H 
