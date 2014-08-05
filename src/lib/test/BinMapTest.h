#ifndef BINMAPTEST_H
#define BINMAPTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file BinMapTest.h
 */

namespace pelican {

namespace ampp {

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
        CPPUNIT_TEST( test_hash );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_bin();
        void test_hash();

    public:
        BinMapTest(  );
        ~BinMapTest();

    private:
};

} // namespace ampp
} // namespace pelican
#endif // BINMAPTEST_H 
