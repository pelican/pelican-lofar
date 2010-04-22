#ifndef LOFARCHUNKERTEST_H
#define LOFARCHUNKERTEST_H

#include <cppunit/extensions/HelperMacros.h>
#include "LofarDataGenerator.h"
#include "pelican/utility/Config.h"

/**
 * @file LofarChunkerTest.h
 */

namespace pelican {
namespace lofar {

/**
 * @class LofarChunkerTest
 *
 * @brief
 *   Unit test for the LofarChunker class
 * @details
 *
 */

class LofarChunkerTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( LofarChunkerTest );
        CPPUNIT_TEST( test_method );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();

    public:
        LofarChunkerTest(  );
        ~LofarChunkerTest();

    private:
        LofarDataGenerator dataGenerator;
        Config config;
};

} // namespace lofar
} // namespace pelican
#endif // LOFARCHUNKERTEST_H
