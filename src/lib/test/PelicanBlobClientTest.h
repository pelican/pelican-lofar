#ifndef PELICANBLOBCLIENTTEST_H
#define PELICANBLOBCLIENTTEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file PelicanBlobClientTest.h
 */

class QCoreApplication;

namespace pelican {

namespace lofar {

/**
 * @class PelicanBlobClientTest
 *  
 * @brief
 * 
 * @details
 * 
 */

class PelicanBlobClientTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( PelicanBlobClientTest );
        CPPUNIT_TEST( test_method );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();

    public:
        PelicanBlobClientTest(  );
        ~PelicanBlobClientTest();

    private:
        QCoreApplication* _app;
};

} // namespace lofar
} // namespace pelican
#endif // PELICANBLOBCLIENTTEST_H 
