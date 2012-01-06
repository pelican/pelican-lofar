#ifndef DEDISPERSIONBUFFERTEST_H
#define DEDISPERSIONBUFFERTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file DedispersionBufferTest.h
 */

namespace pelican {

namespace lofar {

/**
 * @class DedispersionBufferTest
 *  
 * @brief
 * 
 * @details
 * 
 */

class DedispersionBufferTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( DedispersionBufferTest );
        CPPUNIT_TEST( test_sizing );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_sizing();

    public:
        DedispersionBufferTest(  );
        ~DedispersionBufferTest();

    private:
};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONBUFFERTEST_H 
