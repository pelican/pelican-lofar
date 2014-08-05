#ifndef ADAPTER_SUBBAND_TIME_SERIES_TEST_H
#define ADAPTER_SUBBAND_TIME_SERIES_TEST_H

/**
 * @file AdapterTimeSeriesDataSetTest.h
 */

#include <cppunit/extensions/HelperMacros.h>

#include "pelican/utility/ConfigNode.h"

#include <QtCore/QString>

namespace pelican {
namespace ampp {

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
        //CPPUNIT_TEST(test_configuration);
        //CPPUNIT_TEST(test_checkDataFixedPacket);
        //CPPUNIT_TEST(test_checkDataVariablePacket);
        CPPUNIT_TEST(test_deserialise);
        CPPUNIT_TEST(test_deserialise_timing);
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
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

        void test_deserialise_timing();

    private:
        ConfigNode _configXml(const QString& fixedSizePackets,
                unsigned dataBitSize, unsigned udpPacketsPerIteration,
                unsigned samplesPerPacket, unsigned outputChannelsPerSubband,
                unsigned subbandsPerPacket, unsigned nRawPolarisations);

    private:
        bool _verbose;
        ConfigNode _config;
        QString _fixedSizePackets;
        unsigned _dataBitSize;
        unsigned _udpPacketsPerIteration;
        unsigned _samplesPerPacket;
        unsigned _outputChannelsPerSubband;
        unsigned _subbandsPerPacket;
        unsigned _nRawPolarisations;
};


} // namespace ampp
} // namespace pelican
#endif // ADAPTER_SUBBAND_TIME_SERIES_TEST_H
