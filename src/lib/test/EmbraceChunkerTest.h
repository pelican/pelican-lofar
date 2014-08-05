#ifndef EMBRACECHUNKERTEST_H
#define EMBRACECHUNKERTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file EmbraceChunkerTest.h
 */

namespace pelican {

namespace ampp {

/**
 * @class EmbraceChunkerTest
 *  
 * @brief
 * 
 * @details
 * 
 */

class EmbraceChunkerTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( EmbraceChunkerTest );
        CPPUNIT_TEST( test_method );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();

    public:
        EmbraceChunkerTest(  );
        ~EmbraceChunkerTest();

    private:
};

} // namespace ampp
} // namespace pelican
#endif // EMBRACECHUNKERTEST_H 
