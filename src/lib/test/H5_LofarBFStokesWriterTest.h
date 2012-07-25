#ifndef H5_LOFARBFSTOKESWRITERTEST_H
#define H5_LOFARBFSTOKESWRITERTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file H5_LofarBFStokesWriterTest.h
 */

namespace pelican {

namespace lofar {
namespace test {
    class TestDir;
}

/**
 * @class H5_LofarBFStokesWriterTest
 *  
 * @brief
 *    Unit test for the H5_LofarBFStokesWriter
 * @details
 * 
 */

class H5_LofarBFStokesWriterTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( H5_LofarBFStokesWriterTest );
        CPPUNIT_TEST( test_method );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();

    public:
        H5_LofarBFStokesWriterTest(  );
        ~H5_LofarBFStokesWriterTest();

    private:
        test::TestDir* _testDir;
};

} // namespace lofar
} // namespace pelican
#endif // H5_LOFARBFSTOKESWRITERTEST_H 
