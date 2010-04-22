#include "test/AdapterTimeStreamTest.h"

#include "AdapterTimeStream.h"
#include "pelican/utility/ConfigNode.h"
#include "pelican/data/TimeStreamData.h"
#include "LofarUdpHeader.h"

#include <iostream>

#include "pelican/utility/memCheck.h"

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(AdapterTimeStreamTest);


void AdapterTimeStreamTest::setUp()
{
}

void AdapterTimeStreamTest::tearDown()
{
}

/**
 * @details
 * Method to test the adapter configuration.
 */
void AdapterTimeStreamTest::test_configuration()
{
    // Create configuration node.
    unsigned nTimes = 10;
    unsigned dataBytes = 4;
    QString xml = ""
            "<AdapterTimeStream name=\"test\">"
            "	<timeSamples number=\"" + QString::number(nTimes) + "\"/>"
            "	<dataBytes number=\"" + QString::number(dataBytes) + "\"/>"
            "</AdapterTimeStream>";
    ConfigNode configNode(xml);

    // Construct the adapter.
    AdapterTimeStream adapter(configNode);

    // Check configuration.
    CPPUNIT_ASSERT_EQUAL(nTimes, adapter._nTimes);
    CPPUNIT_ASSERT_EQUAL(dataBytes, adapter._dataBytes);
}


/**
 * @details
 * Method to test the _checkData() method of the adapter.
 */
void AdapterTimeStreamTest::test_checkData()
{
    // Create configuration node.
    unsigned nTimes = 10;
    unsigned dataBytes = 4;
    QString xml = ""
            "<AdapterTimeStream name=\"test\">"
            "	<timeSamples number=\"" + QString::number(nTimes) + "\"/>"
            "	<dataBytes number=\"" + QString::number(dataBytes) + "\"/>"
            "</AdapterTimeStream>";
    ConfigNode configNode(xml);

    // Construct the adapter.
    AdapterTimeStream adapter(configNode);

    // Construct a data blob to adapt into.
    TimeStreamData data(nTimes);

    // Set the data blob to be adapted, the input chuck size and associated
    // service data.
    //size_t chunkSize = nTimes * dataBytes * 2;
    unsigned packetSize = sizeof(UDPPacket);
    //size_t chunkSize = nTimes * dataBytes * 2;
    size_t chunkSize = packetSize * 2;
    adapter.config(&data, chunkSize, QHash<QString, DataBlob*>());

    // Check the adapter.config() method behaved as expected.
    CPPUNIT_ASSERT_EQUAL(chunkSize, adapter._chunkSize);
    CPPUNIT_ASSERT_EQUAL(0, adapter._serviceData.size());
    CPPUNIT_ASSERT_EQUAL(nTimes,
            static_cast<TimeStreamData*>(adapter._data)->size());

    //try {
    //   adapter._checkData();
    //}
    //catch (QString err) {
    //    CPPUNIT_FAIL(err.toStdString().data());
    // }

}



} // namespace lofar
} // namespace pelican
