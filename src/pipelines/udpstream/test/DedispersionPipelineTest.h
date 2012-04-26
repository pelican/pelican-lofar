#ifndef DEDISPERSIONPIPELINETEST_H
#define DEDISPERSIONPIPELINETEST_H

#include <cppunit/extensions/HelperMacros.h>
#include <QString>

/**
 * @file DedispersionPipelineTest.h
 */

namespace pelican {

namespace lofar {

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
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();

    public:
        DedispersionPipelineTest(  );
        ~DedispersionPipelineTest();
        QString config( const QString& pipelineConf ) const;

    private:
};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONPIPELINETEST_H 
