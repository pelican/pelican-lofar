#ifndef DEDISPERSIONSPECTRATEST_H
#define DEDISPERSIONSPECTRATEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file DedispersionSpectraTest.h
 */

namespace pelican {

namespace lofar {

/**
 * @class DedispersionSpectraTest
 *  
 * @brief
 *    Unit test for the DedispersionSpectra
 * @details
 * 
 */

class DedispersionSpectraTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( DedispersionSpectraTest );
        CPPUNIT_TEST( test_dmIndex );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_dmIndex();

    public:
        DedispersionSpectraTest(  );
        ~DedispersionSpectraTest();

    private:
};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONSPECTRATEST_H 
