#ifndef DEDISPERSIONPIPELINETEST_H
#define DEDISPERSIONPIPELINETEST_H

#include <cppunit/extensions/HelperMacros.h>
#include <QString>

/**
 * @file DedispersionPipelineTest.h
 */

namespace pelican {

namespace ampp {

/**
 * @class DedispersionPipelineTest
 *  
 * @brief
 * 
 * @details
 * 
 */

class DedispersionPipelineTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( DedispersionPipelineTest );
        CPPUNIT_TEST( test_method );
        CPPUNIT_TEST( test_lofar );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();
        void test_lofar(); // lofar scale test

    public:
        DedispersionPipelineTest(  );
        ~DedispersionPipelineTest();
        QString config( const QString& pipelineConf ) const;

    private:
};

} // namespace ampp
} // namespace pelican
#endif // DEDISPERSIONPIPELINETEST_H 
