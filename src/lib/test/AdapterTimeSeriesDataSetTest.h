#ifndef ADAPTER_SUBBAND_TIME_SERIES_TEST_H
#define ADAPTER_SUBBAND_TIME_SERIES_TEST_H

#include <cppunit/extensions/HelperMacros.h>

/**
 * @file AdapterTimeSeriesDataSetTest.h
 */

#include <QtCore/QString>

namespace pelican {
namespace lofar {

/**
 * @class AdapterTimeSeriesDataSetTest
 *
 * @brief
 * @details
 */

class AdapterTimeSeriesDataSetTest : public CppUnit::TestFixture
{
    public:
        AdapterTimeSeriesDataSetTest() : CppUnit::TestFixture() {}
        ~AdapterTimeSeriesDataSetTest() {}

    public:
        CPPUNIT_TEST_SUITE(AdapterTimeSeriesDataSetTest);
        CPPUNIT_TEST(test_configuration);
        CPPUNIT_TEST(test_checkDataFixedPacket);
        CPPUNIT_TEST(test_checkDataVariablePacket);
        CPPUNIT_TEST(test_deserialise);
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp() {}
        void tearDown() {}

        /// Method to test the adapter configuration.
        void test_configuration();

        /// Test creating with factory.
        void test_factoryCreate();

        /// Method to check data validation performed by the adapter for fixed
        /// packet size.
        void test_checkDataFixedPacket();

        /// Method to check data validation performed by the adapter for variable
        /// packet size.
        void test_checkDataVariablePacket();

        /// Method to check deserialising a chunk of UDP packets.
        void test_deserialise();

    private:
        QString _configXml(const QString& fixedSizePackets,
                unsigned sampleBits, unsigned nPacketsPerChunk,
                unsigned nSamplesPerPacket, unsigned nSamplesPerBlock,
                unsigned nSubbands, unsigned nPolarisations);
};


} // namespace lofar
} // namespace pelican
#endif // ADAPTER_SUBBAND_TIME_SERIES_TEST_H
