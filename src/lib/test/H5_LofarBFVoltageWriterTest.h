#ifndef H5_LOFARBFVOLTAGEWRITERTEST_H
#define H5_LOFARBFVOLTAGEWRITERTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file H5_LofarBFVoltageWriterTest.h
 */

namespace pelican {

namespace lofar {
namespace test {
    class TestDir;
}

/**
 * @class H5_LofarBFVoltageWriterTest
 *  
 * @brief
 * 
 * @details
 * 
 */

class H5_LofarBFVoltageWriterTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( H5_LofarBFVoltageWriterTest );
        CPPUNIT_TEST( test_method );
        CPPUNIT_TEST( test_performance );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();
        void test_performance();

    public:
        H5_LofarBFVoltageWriterTest(  );
        ~H5_LofarBFVoltageWriterTest();

    private:
        test::TestDir* _testDir;
};

} // namespace lofar
} // namespace pelican
#endif // H5_LOFARBFVOLTAGEWRITERTEST_H 
