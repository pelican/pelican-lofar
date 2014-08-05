#ifndef BANDPASSTEST_H
#define BANDPASSTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file BandPassTest.h
 */

namespace pelican {

namespace ampp {

/**
 * @class BandPassTest
 *  
 * @brief
 *   Unit test for the BandPass class
 * @details
 * 
 */

class BandPassTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( BandPassTest );
        CPPUNIT_TEST( test_reBin );
        CPPUNIT_TEST( test_setMedian );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_reBin();
        void test_setMedian();

    public:
        BandPassTest(  );
        ~BandPassTest();

    private:
};

} // namespace ampp
} // namespace pelican
#endif // BANDPASSTEST_H 
