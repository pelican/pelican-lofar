#ifndef DEDISPERSIONMODULETEST_H
#define DEDISPERSIONMODULETEST_H

#include <cppunit/extensions/HelperMacros.h>
#include <QString>
#include "pelican/utility/LockingCircularBuffer.hpp"
#include "DedispersedTimeSeries.h"

/**
 * @file DedispersionModuleTest.h
 */

namespace pelican {
class ConfigNode;
class DataBlob;

namespace lofar {

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
        CPPUNIT_TEST( test_method );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();

        // utility methods
        void connected( DataBlob* dataOut );

    public:
        DedispersionModuleTest(  );
        ~DedispersionModuleTest();

    protected:
        ConfigNode testConfig(QString file = QString() ) const;
        LockingCircularBuffer<DedispersedTimeSeries<float>* >* outputBuffer(int size);
        void destroyBuffer(LockingCircularBuffer<DedispersedTimeSeries<float>* >* b);
        QList<DedispersedTimeSeries<float>* > _outputData;

    private:
        int _connectCount;
        DedispersedTimeSeries<float>* _connectData;
        
};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONMODULETEST_H 
