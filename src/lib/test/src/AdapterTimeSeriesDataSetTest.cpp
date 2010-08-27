#include "test/AdapterTimeSeriesDataSetTest.h"

#include "AdapterTimeSeriesDataSet.h"

#include "pelican/utility/FactoryGeneric.h"

#include "TimeSeriesDataSet.h"
#include "LofarUdpHeader.h"
#include "LofarTypes.h"

#include <QtCore/QBuffer>
#include <QtCore/QDebug>
#include <QtCore/QTime>

#include <vector>
#include <complex>
#include <cstdlib>

#include <iostream>

using std::cout;
using std::endl;

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(AdapterTimeSeriesDataSetTest);


void AdapterTimeSeriesDataSetTest::setUp()
{
    _verbose = true;

    _fixedSizePackets = "true";
    _dataBitSize = 16;
    _udpPacketsPerIteration = 32000;
    _samplesPerPacket = 16;
    _outputChannelsPerSubband = 16;
    _subbandsPerPacket = 61;
    _nRawPolarisations = 2;
}


/**
 * @details
 * Method to test the adapter configuration.
 */
void AdapterTimeSeriesDataSetTest::test_configuration()
{
    _config = _configXml(_fixedSizePackets, _dataBitSize,
            _udpPacketsPerIteration, _samplesPerPacket,
            _outputChannelsPerSubband, _subbandsPerPacket, _nRawPolarisations);

    // Construct the adapter.
    AdapterTimeSeriesDataSet adapter(_config);

    // Check configuration.
    CPPUNIT_ASSERT_EQUAL(true, adapter._fixedPacketSize);
    CPPUNIT_ASSERT_EQUAL(_dataBitSize, adapter._sampleBits);
    CPPUNIT_ASSERT_EQUAL(_udpPacketsPerIteration, adapter._nUDPPacketsPerChunk);
    CPPUNIT_ASSERT_EQUAL(_samplesPerPacket, adapter._nSamplesPerPacket);
    CPPUNIT_ASSERT_EQUAL(_outputChannelsPerSubband, adapter._nSamplesPerTimeBlock);
    CPPUNIT_ASSERT_EQUAL(_subbandsPerPacket, adapter._nSubbands);
    CPPUNIT_ASSERT_EQUAL(_nRawPolarisations, adapter._nPolarisations);
}


/**
 * @details
 * Method to test the _checkData() method of the adapter.
 */
void AdapterTimeSeriesDataSetTest::test_checkDataFixedPacket()
{
    try {
        _config = _configXml(_fixedSizePackets, _dataBitSize,
                _udpPacketsPerIteration, _samplesPerPacket,
                _outputChannelsPerSubband, _subbandsPerPacket, _nRawPolarisations);

        // Construct the adapter.
        AdapterTimeSeriesDataSet adapter(_config);

        // Construct a data blob to adapt into.
        TimeSeriesDataSetC32 timeSeries;

        // Set the data blob to be adapted, the input chuck size and associated
        // service data.
        unsigned packetSize = sizeof(UDPPacket);
        size_t chunkSize = packetSize * _udpPacketsPerIteration;
        adapter.config(&timeSeries, chunkSize, QHash<QString, DataBlob*>());

        // Check the adapter.config() method behaved as expected.
        CPPUNIT_ASSERT_EQUAL(chunkSize, adapter._chunkSize);
        CPPUNIT_ASSERT_EQUAL(0, adapter._serviceData.size());

        adapter._checkData();

        unsigned nTimes = (_udpPacketsPerIteration * _samplesPerPacket);
        unsigned nTimeBlocks = nTimes / _outputChannelsPerSubband;
        unsigned nTimeSeries = nTimeBlocks * _nRawPolarisations * _subbandsPerPacket;

        CPPUNIT_ASSERT_EQUAL(nTimeBlocks, adapter._timeData->nTimeBlocks());
        CPPUNIT_ASSERT_EQUAL(nTimeSeries, adapter._timeData->size());
    }
    catch (QString err) {
        CPPUNIT_FAIL(err.toStdString().data());
    }
}


void AdapterTimeSeriesDataSetTest::test_factoryCreate()
{
//    try {
//        FactoryGeneric<AbstractAdapter> factory;
//        AbstractAdapter* a = factory.create("AdapterTimeSeriesDataSet");
//    }
//    catch (QString err) {
//        CPPUNIT_FAIL(err.toStdString().data());
//    }
}

/**
 * @details
 * Method to test the _checkData() method of the adapter.
 */
void AdapterTimeSeriesDataSetTest::test_checkDataVariablePacket()
{
    try {
        // Create configuration node.
        _fixedSizePackets = "false";
        _config = _configXml(_fixedSizePackets, _dataBitSize,
                _udpPacketsPerIteration, _samplesPerPacket,
                _outputChannelsPerSubband, _subbandsPerPacket, _nRawPolarisations);

        // Construct the adapter.
        AdapterTimeSeriesDataSet adapter(_config);

        // Construct a data blob to adapt into.
        TimeSeriesDataSetC32 timeSeries;

        // Set the data blob to be adapted, the input chuck size and associated
        // service data.
        unsigned nTimes = (_udpPacketsPerIteration * _samplesPerPacket);
        unsigned nTimeBlocks = nTimes / _outputChannelsPerSubband;
        unsigned nTimeSeries = nTimeBlocks * _nRawPolarisations * _subbandsPerPacket;

        unsigned nData = _subbandsPerPacket * _nRawPolarisations * _samplesPerPacket;
        size_t packetSize = sizeof(UDPPacket::Header) + (nData * _dataBitSize * 2) / 8;
        size_t chunkSize = packetSize * _udpPacketsPerIteration;
        adapter.config(&timeSeries, chunkSize, QHash<QString, DataBlob*>());

        // Check the adapter.config() method behaved as expected.
        CPPUNIT_ASSERT_EQUAL(chunkSize, adapter._chunkSize);
        CPPUNIT_ASSERT_EQUAL(0, adapter._serviceData.size());

        adapter._checkData();
        CPPUNIT_ASSERT_EQUAL(nTimeBlocks, adapter._timeData->nTimeBlocks());
        CPPUNIT_ASSERT_EQUAL(nTimeSeries, adapter._timeData->size());

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
    try {
        // Create configuration node.
        _config = _configXml(_fixedSizePackets, _dataBitSize,
                _udpPacketsPerIteration, _samplesPerPacket,
                _outputChannelsPerSubband, _subbandsPerPacket, _nRawPolarisations);

        typedef TYPES::i8complex i8c;
        typedef TYPES::i16complex i16c;

        // Construct the adapter.
        AdapterTimeSeriesDataSet adapter(_config);

        // Construct a data blob to adapt into.
        TimeSeriesDataSetC32 timeSeries;

        size_t chunkSize = sizeof(UDPPacket) * _udpPacketsPerIteration;

        // Configure the adapter setting the data blob, chunk size and service data.
        adapter.config(&timeSeries, chunkSize, QHash<QString, DataBlob*>());

        // Create and fill a UDP packet.
        std::vector<UDPPacket> packets(_udpPacketsPerIteration);
        unsigned index = 0;
        for (unsigned i = 0; i < _udpPacketsPerIteration; ++i) {

            // Fill in the header
            packets[i].header.version             = uint8_t(0 + i);
            packets[i].header.sourceInfo          = uint8_t(1 + i);
            packets[i].header.configuration       = uint16_t(_dataBitSize);
            packets[i].header.station             = uint16_t(3 + i);
            packets[i].header.nrBeamlets          = uint8_t(4 + i);
            packets[i].header.nrBlocks            = uint8_t(5 + i);
            packets[i].header.timestamp           = uint32_t(6 + i);
            packets[i].header.blockSequenceNumber = uint32_t(7 + i);

            // Fill in the data
            for (unsigned ii = 0, t = 0; t < _samplesPerPacket; ++t) {
                for (unsigned c = 0; c < _subbandsPerPacket; ++c) {
                    for (unsigned p = 0; p < _nRawPolarisations; ++p) {

                        if (_dataBitSize == 8) {
                            i8c* data = reinterpret_cast<i8c*>(packets[i].data);
                            index = _nRawPolarisations * (t * _subbandsPerPacket + c) + p;
                            data[index] = i8c(ii++, i);
                        }
                        else if (_dataBitSize == 16) {
                            i16c* data = reinterpret_cast<i16c*>(packets[i].data);
                            index = _nRawPolarisations * (t * _subbandsPerPacket + c) + p;
                            data[index] = i16c(ii++, i);
                        }

                    }
                }
            }
        }


        // Stick the packet into an QIODevice.
        QBuffer buffer;
        buffer.setData(reinterpret_cast<char*>(&packets[0]), chunkSize);
        buffer.open(QBuffer::ReadOnly);


        adapter.deserialise(&buffer);
    }
    catch (QString err) {
        CPPUNIT_FAIL(err.toStdString().data());
    }
}


void AdapterTimeSeriesDataSetTest::test_deserialise_timing()
{
    try {
        // Create configuration node.
        _fixedSizePackets = "false";
        _config = _configXml(_fixedSizePackets, _dataBitSize,
                _udpPacketsPerIteration, _samplesPerPacket,
                _outputChannelsPerSubband, _subbandsPerPacket, _nRawPolarisations);

        typedef TYPES::i16complex i16c;

        // Construct the adapter.
        AdapterTimeSeriesDataSet adapter(_config);

        // Construct a data blob to adapt into.
        TimeSeriesDataSetC32 timeSeries;

        unsigned nTimes = (_udpPacketsPerIteration * _samplesPerPacket);
        unsigned nTimeBlocks = nTimes / _outputChannelsPerSubband;
        unsigned nTimeSeries = nTimeBlocks * _nRawPolarisations * _subbandsPerPacket;

        unsigned nData = _subbandsPerPacket * _nRawPolarisations * _samplesPerPacket;
        size_t packetSize = sizeof(UDPPacket::Header) + (nData * _dataBitSize * 2) / 8;
        size_t chunkSize = packetSize * _udpPacketsPerIteration;

        // Configure the adapter setting the data blob, chunk size and service data.
        adapter.config(&timeSeries, chunkSize, QHash<QString, DataBlob*>());

        // Create and fill a UDP packet.
        std::vector<UDPPacket> packets(_udpPacketsPerIteration);
        unsigned index = 0;
        for (unsigned i = 0; i < _udpPacketsPerIteration; ++i) {

            // Fill in the header
            packets[i].header.version             = uint8_t(0 + i);
            packets[i].header.sourceInfo          = uint8_t(1 + i);
            packets[i].header.configuration       = uint16_t(_dataBitSize);
            packets[i].header.station             = uint16_t(3 + i);
            packets[i].header.nrBeamlets          = uint8_t(4 + i);
            packets[i].header.nrBlocks            = uint8_t(5 + i);
            packets[i].header.timestamp           = uint32_t(6 + i);
            packets[i].header.blockSequenceNumber = uint32_t(7 + i);

            // Fill in the data
            for (unsigned ii = 0, t = 0; t < _samplesPerPacket; ++t) {
                for (unsigned c = 0; c < _subbandsPerPacket; ++c) {
                    for (unsigned p = 0; p < _nRawPolarisations; ++p) {
                        i16c* data = reinterpret_cast<i16c*>(packets[i].data);
                        index = _nRawPolarisations * (t * _subbandsPerPacket + c) + p;
                        data[index] = i16c(ii++, i);
                    }
                }
            }
        }


        // Stick the packet into an QIODevice.
        QBuffer buffer;
        buffer.setData(reinterpret_cast<char*>(&packets[0]), chunkSize);
        buffer.open(QBuffer::ReadOnly);

        QTime timer;
        timer.start();
        adapter.deserialise(&buffer);
        int elapsed = timer.elapsed();
        cout << endl;
        cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
        cout << "[AdapterTimeSeriesDataSet]: deserialise() " << endl;
        cout << "- nChan = " << _outputChannelsPerSubband << endl << endl;
        if (_verbose) {
            cout << "- nBlocks = " << nTimeBlocks << endl;
            cout << "- nSubbands = " << _subbandsPerPacket << endl;
            cout << "- nPols = " << _nRawPolarisations << endl;
            cout << "- nTimes = " << nTimes << endl;
        }
        cout << "* Elapsed = " << elapsed << " ms." << endl;
        cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;

    }
    catch (QString err) {
        CPPUNIT_FAIL(err.toStdString().data());
    }
}








/**
 * @details
 * Construct a config node for use with the adapter.
 */
ConfigNode AdapterTimeSeriesDataSetTest::_configXml(
        const QString& fixedSizePackets, unsigned dataBitSize,
        unsigned udpPacketsPerIteration, unsigned samplesPerPacket,
        unsigned outputChannelsPerSubband, unsigned subbandsPerPacket,
        unsigned nRawPolarisations)
{
    QString xml =
            "<AdapterSubbandTimeSeries name=\"test\">"
            "   <fixedSizePackets         value=\"%1\"/>"
            "   <dataBitSize              value=\"%2\"/>"
            "   <udpPacketsPerIteration   value=\"%3\"/>"
            "   <samplesPerPacket         value=\"%4\"/>"
            "   <outputChannelsPerSubband value=\"%5\"/>"
            "   <subbandsPerPacket        value=\"%6\"/>"
            "   <nRawPolarisations        value=\"%7\"/>"
            "</AdapterSubbandTimeSeries>";

    xml = xml.arg(fixedSizePackets);
    xml = xml.arg(dataBitSize);
    xml = xml.arg(udpPacketsPerIteration);
    xml = xml.arg(samplesPerPacket);
    xml = xml.arg(outputChannelsPerSubband);
    xml = xml.arg(subbandsPerPacket);
    xml = xml.arg(nRawPolarisations);

    return ConfigNode(xml);
}


} // namespace lofar
} // namespace pelican
