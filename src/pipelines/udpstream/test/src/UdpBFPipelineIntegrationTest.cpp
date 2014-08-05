#include "UdpBFPipelineIntegrationTest.h"
#include "pelican/emulator/EmulatorDriver.h"
#include "pelican/utility//ConfigNode.h"
#include "LofarTestClient.h"
#include "test/LofarEmulatorDataSim.h"
#include "testInfo.h"
#include "TimeSeriesDataSet.h"
#include "AdapterTimeSeriesDataSet.h"
#include <QProcess>
#include <QFile>
#include <QList>
#include <QString>
#include <QCoreApplication>


namespace pelican {

namespace ampp {

CPPUNIT_TEST_SUITE_REGISTRATION( UdpBFPipelineIntegrationTest );
/**
 *@details UdpBFPipelineIntegrationTest 
 */
UdpBFPipelineIntegrationTest::UdpBFPipelineIntegrationTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
UdpBFPipelineIntegrationTest::~UdpBFPipelineIntegrationTest() {
}

void UdpBFPipelineIntegrationTest::setUp() {
    _server = new QProcess();
    _server->setProcessChannelMode(QProcess::ForwardedChannels);
}

void UdpBFPipelineIntegrationTest::tearDown() {
    delete _server;
}

void UdpBFPipelineIntegrationTest::startServer() {
    QList<QString> args;
    args << "--config" << pelican::ampp::test::TEST_DATA_DIR + "/integrationServerConfig.xml";
    _server->start( pelican::ampp::test::SERVER_EXECUTABLE , args );
    if (! _server->waitForStarted()) {
        CPPUNIT_FAIL("Cannot start server");
    }
}

void UdpBFPipelineIntegrationTest::stopServer() {
    _server->terminate();
    if (! _server->waitForFinished()) {
        CPPUNIT_FAIL("Cannot stop server");
    }
}


void UdpBFPipelineIntegrationTest::test_topdownInit()
{
    QString stream1 = "LofarTimeStream1";
    QString stream2 = "LofarTimeStream2";
    unsigned long clientCalledCount1 = 0;
    unsigned long bufferSize = 1000;
    ConfigNode emulatorConfig;
    emulatorConfig.setFromString(
            "<LofarEmulatorDataSim>"
                "<connection host=\"127.0.0.1\" port=\"4347\"/>"
                "    <packetSendInterval value=\"200\"/>"
                "    <packetStartDelay   value=\"1\"/>"
                "    <polsPerPacket      value=\"2\"/>"
                "    <subbandsPerPacket value=\"31\"/> <!-- 31 or 61 or 62 -->"
                "    <samplesPerPacket value=\"16\" />"
                "    <clock value=\"200\" /> <!-- Could also be 160 -->"
                "<dataBitSize value=\"16\" />"
                "<fixedSizePackets value=\"false\" />"
                "<outputChannelsPerSubband value=\"32\" />"
                "<udpPacketsPerIteration value=\"64\" />"
                "</LofarEmulatorDataSim>"
            );
    try {
        LofarEmulatorDataSim* emulator = new LofarEmulatorDataSim(emulatorConfig);
        QFile config( pelican::ampp::test::TEST_DATA_DIR + "/integrationConfig.xml" );
        LofarTestClient client1(emulator, config, stream1);
        LofarTestClient client2(emulator, config, stream2);
        EmulatorDriver data(emulator); // takes ownership of the emulator

        // Use Case
        // Instantiate a data stream, a packet splitting server,
        // and a suitable Client
        // Expect:
        // Everything to stay up and data to be propagated down the chain
        //
        data.start();
        CPPUNIT_ASSERT_EQUAL( (unsigned long)0, client1.count() );
        CPPUNIT_ASSERT_EQUAL( (unsigned long)0, client2.count() );
        startServer();
        client1.startup();
        client2.startup();
        while( ! client1.count() ) { QCoreApplication::processEvents(QEventLoop::WaitForMoreEvents, 10); }
        clientCalledCount1 = client1.count();
        CPPUNIT_ASSERT( clientCalledCount1 >= 1 );

        // Use Case
        // Kill the data stream, wait and then restart
        // Expect
        // Server and client to wait patiently, and restart processing
        // when the data stream comes back online
        //
        data.terminate();
        data.wait();
        clientCalledCount1 = client1.count();
        data.start();
        do { QCoreApplication::processEvents( QEventLoop::WaitForMoreEvents, 10); }
        while( !( clientCalledCount1 < client1.count()) );
        clientCalledCount1 = client1.count();

        // Use Case
        //  Kill the server, wait and then restart
        //  Expect
        //  Client to wait patiently, and restart processing
        //  when the server comes back online
        //
        stopServer();
        startServer();
        sleep(1);
        CPPUNIT_ASSERT( _server->pid() != 0 );
        do { QCoreApplication::processEvents( QEventLoop::WaitForMoreEvents, 10); }
        while( !( clientCalledCount1 < client1.count()) );
        clientCalledCount1 = client1.count();

        // Use Case
        //  Kill one client, and fill the server buffer, before
        //  restarting
        // Expect
        //  Client to continue processing data
        //
        client1.exit();
        unsigned long start = data.dataCount();
        do{ sleep(1); }
        while( ( data.dataCount() - start ) < bufferSize );
        client1.start();
        do { QCoreApplication::processEvents(); usleep(5); }
        while( !( clientCalledCount1 < client1.count()) );
        clientCalledCount1 = client1.count();
        data.terminate();
     }
     catch ( const QString& e ) {
        stopServer();
        CPPUNIT_FAIL( "Caught Signal: " + e.toStdString() );
     }
     stopServer();
}

} // namespace ampp
} // namespace pelican
