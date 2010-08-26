#include "test/AdapterTimeSeriesDataSetTest.h"

#include "AdapterTimeSeriesDataSet.h"

#include "pelican/utility/FactoryGeneric.h"

#include "pelican/utility/ConfigNode.h"
#include "TimeSeriesDataSet.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"

#include <QtCore/QBuffer>

#include <vector>
#include <complex>
#include <cstdlib>

#include <iostream>

#include "pelican/utility/memCheck.h"

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(AdapterTimeSeriesDataSetTest);


/**
 * @details
 * Method to test the adapter configuration.
 */
void AdapterTimeSeriesDataSetTest::test_configuration()
{
    // Create configuration node.
    QString fixedSizePackets = "true";
    unsigned sampleBits = 8;
    unsigned nPackets = 1;
    unsigned nSamples = 10;
    unsigned nSamplesPerTimeBlock = 512;
    unsigned nSubbands = 1;
    unsigned nPolarisations = 1;
    QString xml = _configXml(fixedSizePackets, sampleBits, nPackets, nSamples,
            nSamplesPerTimeBlock, nSubbands, nPolarisations);
    ConfigNode configNode(xml);

    // Construct the adapter.
    AdapterTimeSeriesDataSet adapter(configNode);

    // Check configuration.
    CPPUNIT_ASSERT_EQUAL(true, adapter._fixedPacketSize);
    CPPUNIT_ASSERT_EQUAL(sampleBits, adapter._sampleBits);
    CPPUNIT_ASSERT_EQUAL(nPackets, adapter._nUDPPacketsPerChunk);
    CPPUNIT_ASSERT_EQUAL(nSamples, adapter._nSamplesPerPacket);
    CPPUNIT_ASSERT_EQUAL(nSamplesPerTimeBlock, adapter._nSamplesPerTimeBlock);
    CPPUNIT_ASSERT_EQUAL(nSubbands, adapter._nSubbands);
    CPPUNIT_ASSERT_EQUAL(nPolarisations, adapter._nPolarisations);
}


/**
 * @details
 * Method to test the _checkData() method of the adapter.
 */
void AdapterTimeSeriesDataSetTest::test_checkDataFixedPacket()
{
    // Create configuration node.
    QString fixedSizePackets = "true";
    unsigned sampleBits = 8;
    unsigned nPackets = 32;
    unsigned nSamples = 32;
    unsigned nSamplesPerTimeBlock = 512;
    unsigned nSubbands = 4;
    unsigned nPolarisations = 2;
    QString xml = _configXml(fixedSizePackets, sampleBits, nPackets, nSamples,
            nSamplesPerTimeBlock, nSubbands, nPolarisations);
    ConfigNode configNode(xml);

    // Construct the adapter.
    AdapterTimeSeriesDataSet adapter(configNode);

    // Construct a data blob to adapt into.
    TimeSeriesDataSetC32 timeSeries;

    // Set the data blob to be adapted, the input chuck size and associated
    // service data.
    unsigned packetSize = sizeof(UDPPacket);
    size_t chunkSize = packetSize * nPackets;
    adapter.config(&timeSeries, chunkSize, QHash<QString, DataBlob*>());

    // Check the adapter.config() method behaved as expected.
    CPPUNIT_ASSERT_EQUAL(chunkSize, adapter._chunkSize);
    CPPUNIT_ASSERT_EQUAL(0, adapter._serviceData.size());

    try {
       adapter._checkData();

       unsigned nTimeBlocks = (nPackets * nSamples) / nSamplesPerTimeBlock;
        CPPUNIT_ASSERT_EQUAL(nTimeBlocks,
                (TimeSeriesDataSetC32*)(adapter._timeData)->nTimeBlocks());
        CPPUNIT_ASSERT_EQUAL(nTimeBlocks * nPolarisations * nSubbands,
                (TimeSeriesDataSetC32*)(adapter._data)->size());
    }
    catch (QString err) {
        CPPUNIT_FAIL(err.toStdString().data());
    }
}


void AdapterTimeSeriesDataSetTest::test_factoryCreate()
{
//    FactoryGeneric<AbstractAdapter> factory;
//    AbstractAdapter* a = factory.create("AdapterSubbandTimeSeries");
}

/**
 * @details
 * Method to test the _checkData() method of the adapter.
 */
void AdapterTimeSeriesDataSetTest::test_checkDataVariablePacket()
{
    // Create configuration node.
    QString fixedSizePackets = "false";
    unsigned sampleBits = 8;
    unsigned nPackets = 32;
    unsigned nSamples = 32;
    unsigned nSamplesPerTimeBlock = 512;
    unsigned nSubbands = 4;
    unsigned nPolarisations = 2;
    QString xml = _configXml(fixedSizePackets, sampleBits, nPackets, nSamples,
            nSamplesPerTimeBlock, nSubbands, nPolarisations);
    ConfigNode configNode(xml);

    // Construct the adapter.
    AdapterTimeSeriesDataSet adapter(configNode);

    // Construct a data blob to adapt into.
    TimeSeriesDataSetC32 timeSeries;

    // Set the data blob to be adapted, the input chuck size and associated
    // service data.
    size_t packetSize = sizeof(UDPPacket::Header) +
            (nSubbands * nPolarisations * nSamples * sampleBits * 2) / 8;
    size_t chunkSize = packetSize * nPackets;
    adapter.config(&timeSeries, chunkSize, QHash<QString, DataBlob*>());

    // Check the adapter.config() method behaved as expected.
    CPPUNIT_ASSERT_EQUAL(chunkSize, adapter._chunkSize);
    CPPUNIT_ASSERT_EQUAL(0, adapter._serviceData.size());

    try {
        adapter._checkData();
        unsigned nTimeBlocks = (nPackets * nSamples) / nSamplesPerTimeBlock;
        CPPUNIT_ASSERT_EQUAL(nTimeBlocks,
                (TimeSeriesDataSetC32*)(adapter._timeData)->nTimeBlocks());
        CPPUNIT_ASSERT_EQUAL(nTimeBlocks * nPolarisations * nSubbands,
                (TimeSeriesDataSetC32*)(adapter._data)->size());
    }
    catch (QString err) {
        CPPUNIT_FAIL(err.toStdString().data());
    }
}


/**
 * @details
 * Method to test the deserialise() method of the adapter.
 */
void AdapterTimeSeriesDataSetTest::test_deserialise()
{
    // Create configuration node.
    QString fixedSizePackets = "true";
    unsigned sampleBits = 8;
    unsigned nPackets = 32;
    unsigned nSamples = 32;
    unsigned nSamplesPerTimeBlock = 512;
    unsigned nSubbands = 4;
    unsigned nPolarisations = 2;
    QString xml = _configXml(fixedSizePackets, sampleBits, nPackets, nSamples,
            nSamplesPerTimeBlock, nSubbands, nPolarisations);
    ConfigNode configNode(xml);

    typedef TYPES::i8complex iComplex16;
    typedef TYPES::i16complex iComplex32;

    // Construct the adapter.
    AdapterTimeSeriesDataSet adapter(configNode);

    // Construct a data blob to adapt into.
    TimeSeriesDataSetC32 timeSeries;

    unsigned packetSize = sizeof(UDPPacket);
    size_t chunkSize = packetSize * nPackets;

    // Configure the adapter setting the data blob, chunk size and service data.
    adapter.config(&timeSeries, chunkSize, QHash<QString, DataBlob*>());

    // Create and fill a UDP packet.
    std::vector<UDPPacket> packets(nPackets);
    for (unsigned i = 0; i < nPackets; ++i) {

        // Fill in the header
        packets[i].header.version             = uint8_t(0 + i);
        packets[i].header.sourceInfo          = uint8_t(1 + i);
        packets[i].header.configuration       = uint16_t(sampleBits);
        packets[i].header.station             = uint16_t(3 + i);
        packets[i].header.nrBeamlets          = uint8_t(4 + i);
        packets[i].header.nrBlocks            = uint8_t(5 + i);
        packets[i].header.timestamp           = uint32_t(6 + i);
        packets[i].header.blockSequenceNumber = uint32_t(7 + i);

        // Fill in the data
        for (unsigned ii = 0, t = 0; t < nSamples; ++t) {
            for (unsigned c = 0; c < nSubbands; ++c) {
                for (unsigned p = 0; p < nPolarisations; ++p) {

                    if (sampleBits == 8) {
                        iComplex16* data = reinterpret_cast<iComplex16*>(packets[i].data);
                        int index = nPolarisations * (t * nSubbands + c) + p;
                        data[index] = iComplex16(ii, i);
                        ++ii;
                    }
                    else if (sampleBits == 16) {
                        iComplex32* data = reinterpret_cast<iComplex32*>(packets[i].data);
                        int index = nPolarisations * (t * nSubbands + c) + p;
                        data[index] = iComplex32(ii, i);
                        ++ii;
                    }

                }
            }
        }
    }


    // Stick the packet into an QIODevice.
    QBuffer buffer;
    buffer.setData(reinterpret_cast<char*>(&packets[0]), chunkSize);
    buffer.open(QBuffer::ReadOnly);

    try {
        adapter.deserialise(&buffer);
    }
    catch (QString err) {
        CPPUNIT_FAIL(err.toStdString().data());
    }
}




QString AdapterTimeSeriesDataSetTest::_configXml(
        const QString& fixedSizePackets, unsigned sampleBits,
        unsigned nPacketsPerChunk, unsigned nSamplesPerPacket,
        unsigned nSamplesPerBlock, unsigned nSubbands,
        unsigned nPolarisations)
{
    QString xml = ""
            "<AdapterSubbandTimeSeries name=\"test\">"
            "   <fixedSizePackets value=\"" + fixedSizePackets + "\"/>"
            "   <dataBitSize value=\"" + QString::number(sampleBits) + "\"/>"
            "   <udpPacketsPerIteration value=\"" + QString::number(nPacketsPerChunk) + "\"/>"
            "   <samplesPerPacket value=\"" + QString::number(nSamplesPerPacket) + "\"/>"
            "   <outputChannelsPerSubband value=\"" + QString::number(nSamplesPerBlock) + "\"/>"
            "   <subbandsPerPacket value=\"" + QString::number(nSubbands) + "\"/>"
            "   <nRawPolarisations value=\"" + QString::number(nPolarisations) + "\"/>"
            "</AdapterSubbandTimeSeries>";
    return xml;
}


} // namespace lofar
} // namespace pelican
