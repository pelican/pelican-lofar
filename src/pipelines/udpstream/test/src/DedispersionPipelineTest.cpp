#include "test/LofarPipelineTester.h"
#include "DedispersionPipelineTest.h"
#include "DedispersionPipeline.h"
#include "TimeSeriesDataSet.h"
#include <QDir>
#include <QDebug>
#define QUOTE(str) #str
#define EXPAND_AND_QUOTE(str) QUOTE(str)
#define TEST_DATA_DIR EXPAND_AND_QUOTE(TEST_DATA)

namespace pelican {

namespace ampp {

CPPUNIT_TEST_SUITE_REGISTRATION( DedispersionPipelineTest );
/**
 *@details DedispersionPipelineTest 
 */
DedispersionPipelineTest::DedispersionPipelineTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
DedispersionPipelineTest::~DedispersionPipelineTest()
{
}

void DedispersionPipelineTest::setUp()
{
}

void DedispersionPipelineTest::tearDown()
{
}

void DedispersionPipelineTest::test_method()
{
     // Use Case:
     // Simple run through 
     // Expect:
     // no segfaults
     int history = 10;
     int sampleSize=2048;
     int numberOfBuffers=2;
     QString streamId = "LofarTimeStream1";
     QString xml = QString("<pipelineConfig>"
                 "   <DedispersionPipeline>"
                 "       <history value=\"%7\" />"
                 "   </DedispersionPipeline>"
                 "</pipelineConfig>"
                 "<modules>"
                     "<PPFChanneliser>"
                        "<outputChannelsPerSubband value=\"16\" />"
                        "<processingThreads value=\"6\"/>"
                        "<filter nTaps=\"8\" filterWindow=\"kaiser\"/>"
                     "</PPFChanneliser>"
                     "<RFI_Clipper active=\"true\" rejectionFactor=\"10.0\" >"
                         "<BandPassData file=\"%1\" />"
                         "<Band matching=\"true\" />"
                         "<History maximum=\"10000\" />"
                     "</RFI_Clipper>"
                     "<DedispersionModule>"
                         "<sampleNumber value=\"%2\" />"
                         "<frequencyChannel1 MHz=\"%3\"/>"
                         "<sampleTime seconds=\"%4\"/>"
                         "<channelBandwidth MHz=\"%5\"/>"
                         "<dedispersionSamples value=\"%6\" />"
                         "<dedispersionStepSize value=\"0.1\" />"
                         "<numberOfBuffers value=\"%8\" />"
                     "</DedispersionModule>"
                 "</modules>")
                  .arg( QString(TEST_DATA_DIR) + QDir::separator() + "band31.bp")
                  .arg( sampleSize / 4 )
                  .arg( 150 )
                  .arg( 0.1 )
                  .arg( -0.2 )
                  .arg ( 100 ).arg( history )
                  .arg( numberOfBuffers );
     try {
         DedispersionPipeline p(streamId);
         LofarPipelineTester tester(&p, config(xml));
         tester.run();
         // multiple runs to ensure the history
         // is full
         // Expect: not to freeze waiting for history
         // buffer to be freed
         tester.run( (history*numberOfBuffers)+2 );
      }
      catch( const QString& e ) {
        CPPUNIT_FAIL( e.toStdString() );
      }
}

void DedispersionPipelineTest::test_lofar()
{
    // Use Case:
    // test with a typical lofar configuration file
    int history = 406;
    int nSubbands = 64;
    int samplesPerPacket = 16;
    QString streamId = "LofarTimeStream1";
    QString commonXML = QString(
         "<samplesPerPacket value=\"%2\" />"
         "<nRawPolarisations value=\"2\" />"
         "<dataBitSize value=\"16\" />"
         "<totalComplexSubbands value=\"512\" />"

         "<clock value=\"200\" /> <!-- Could also be 160 -->"
         "<outputChannelsPerSubband value=\"%1\" />"
         "<udpPacketsPerIteration value=\"128\" />"
         "<integrateTimeBins value=\"16\" />"
    ).arg( nSubbands ).arg( samplesPerPacket );
    QString xml = QString(
                 "<pipelineConfig>"
                 "   <DedispersionPipeline>"
                 "       <history value=\"%2\" />"
                 "   </DedispersionPipeline>"
                 "</pipelineConfig>"
                 "<modules>"
                 "      <PPFChanneliser>"
                 "           %1"
                 "           <processingThreads value=\"6\" />"
                 "           <filter nTaps=\"8\" filterWindow=\"kaiser\"/>"
                 "      </PPFChanneliser>"
                 "      <StokesGenerator>"
                 "      </StokesGenerator>"

                 "      <RFI_Clipper active=\"true\" channelRejectionRMS=\"10.0\""
                 "                   spectrumRejectionRMS=\"6.0\">"
                 "        <zeroDMing active=\"true\" />"
                 "        <BandPassData file=\"%3\" />"
                 "        <Band matching=\"true\" />"
                 "        <History maximum=\"10000\" />"
                 "      </RFI_Clipper>"
                 "      <DedispersionModule>"
                 "         <sampleNumber value=\"8192\" />"
                 "         <frequencyChannel1 MHz=\"148.828125\"/>"
                 "         <sampleTime seconds=\"0.0032768\"/>"
                 "         <channelBandwidth MHz=\"-0.003051757812\"/>"
                 "         <dedispersionSamples value=\"1000\" />"
                 "         <dedispersionStepSize value=\"0.1\" />"
                 "         <numberOfBuffers value=\"2\" />"
                 "      </DedispersionModule>"
                 "      <StokesIntegrator>"
                 "        %1"
                 "      </StokesIntegrator>"
                 "    </modules>"
        ).arg(commonXML)
         .arg(history)
         .arg( QString(TEST_DATA_DIR) + QDir::separator() + "band31.bp");
    try {
        DedispersionPipeline p(streamId);
        LofarPipelineTester tester(&p, config(xml));
        tester.run( history * 2);
    }
    catch( const QString& e ) {
       CPPUNIT_FAIL( e.toStdString() );
    }
}

QString DedispersionPipelineTest::config( const QString& pipelineConf ) const {
    QString s = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                "<!DOCTYPE pelican>\n\n<configuration version=\"1.0\">"
                "<pipeline>";
     return s + pipelineConf + QString("</pipeline>"
                           "</configuration>\n"
                          );
}

} // namespace ampp
} // namespace pelican
