#ifndef LOCKINGCONTAINERTEST_H
#define LOCKINGCONTAINERTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file LockingContainerTest.h
 */

namespace pelican {

namespace lofar {

/**
 * @class LockingContainerTest
 *  
 * @brief
 *    Unit test for the LockingContainer template class
 * @details
 * 
 */

class LockingContainerTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( LockingContainerTest );
        CPPUNIT_TEST( test_method );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();

    public:
        LockingContainerTest(  );
        ~LockingContainerTest();

    private:
};

} // namespace lofar
} // namespace pelican
#endif // LOCKINGCONTAINERTEST_H 
