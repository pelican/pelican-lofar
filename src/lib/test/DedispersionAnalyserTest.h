#ifndef DEDISPERSIONANALYSERTEST_H
#define DEDISPERSIONANALYSERTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file DedispersionAnalyserTest.h
 */

namespace pelican {

namespace lofar {

/**
 * @class DedispersionAnalyserTest
 *  
 * @brief
 * 
 * @details
 * 
 */

class DedispersionAnalyserTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( DedispersionAnalyserTest );
        CPPUNIT_TEST( test_noSignificantEvents );
        CPPUNIT_TEST( test_singleEvent );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_singleEvent();
        void test_noSignificantEvents();

    public:
        DedispersionAnalyserTest(  );
        ~DedispersionAnalyserTest();

    private:
};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONANALYSERTEST_H 
