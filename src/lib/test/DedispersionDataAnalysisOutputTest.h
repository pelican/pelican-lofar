#ifndef DEDISPERSIONDATAANALYSISOUTPUTTEST_H
#define DEDISPERSIONDATAANALYSISOUTPUTTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file DedispersionDataAnalysisOutputTest.h
 */

namespace pelican {

namespace lofar {

/**
 * @class DedispersionDataAnalysisOutputTest
 *  
 * @brief
 * 
 * @details
 * 
 */

class DedispersionDataAnalysisOutputTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( DedispersionDataAnalysisOutputTest );
        CPPUNIT_TEST( test_method );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();

    public:
        DedispersionDataAnalysisOutputTest(  );
        ~DedispersionDataAnalysisOutputTest();

    private:
};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONDATAANALYSISOUTPUTTEST_H 
