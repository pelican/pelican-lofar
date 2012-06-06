#ifndef DEDISPERSIONMODULETEST_H
#define DEDISPERSIONMODULETEST_H

#include <cppunit/extensions/HelperMacros.h>
#include <QString>
#include "pelican/utility/LockingCircularBuffer.hpp"
#include "LockingPtrContainer.hpp"
#include "DedispersionSpectra.h"

/**
 * @file DedispersionModuleTest.h
 */

namespace pelican {
class ConfigNode;
class DataBlob;

namespace lofar {
class SpectrumDataSetStokes;

/**
 * @class DedispersionModuleTest
 *  
 * @brief
 *    Dedispersion Pipeline Unit Test
 * @details
 * 
 */

class DedispersionModuleTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( DedispersionModuleTest );
        CPPUNIT_TEST( test_multipleBlobsPerBuffer );
        CPPUNIT_TEST( test_method );
        CPPUNIT_TEST( test_multipleBlobs );
        CPPUNIT_TEST( test_multipleBuffersPerBlob );
        CPPUNIT_TEST( test_multipleBlobsPerBufferUnaligned );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();
        void test_multipleBlobs();
        void test_multipleBlobsPerBuffer();
        void test_multipleBlobsPerBufferUnaligned();
        void test_multipleBuffersPerBlob();

        // utility methods
        void connected( DataBlob* dataOut );
        void connectFinished();
        void unlockCallback( const QList<DataBlob*>&  );

    public:
        DedispersionModuleTest(  );
        ~DedispersionModuleTest();

    protected:
        ConfigNode testConfig(QString file = QString() ) const;

    private:
        int _connectCount;
        DedispersionSpectra* _connectData;
        int _chainFinished;
        QList<DataBlob*> _unlocked;
};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONMODULETEST_H 
