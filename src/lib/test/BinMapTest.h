#ifndef BINMAPTEST_H
#define BINMAPTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file BinMapTest.h
 */

namespace pelican {

namespace lofar {

/**
 * @class BinMapTest
 *  
 * @brief
 *    Unit test for BinMap class
 * @details
 * 
 */

class BinMapTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( BinMapTest );
        CPPUNIT_TEST( test_bin );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_bin();

    public:
        BinMapTest(  );
        ~BinMapTest();

    private:
};

} // namespace lofar
} // namespace pelican
#endif // BINMAPTEST_H 
