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

namespace lofar {

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
     // Single run through 
     // Expect:
     // no segfaults
     QString streamId = "LofarTimeStream1";
     QString xml = QString("<pipelineConfig />"
                 "<modules>"
                     "<RFI_Clipper active=\"true\" rejectionFactor=\"10.0\" >"
                     "<BandPassData file=\"%1\" />"
                     "<Band matching=\"true\" />"
                     "<History maximum=\"10000\" />"
                     "</RFI_Clipper>"
                     "<DedispersionModule>"
                     " <sampleNumber value=\"%2\" />"
                     " <frequencyChannel1 value=\"%3\"/>"
                     " <sampleTime value=\"%4\"/>"
                     " <channelBandwidth value=\"%5\"/>"
                     " <dedispersionSamples value=\"%6\" />"
                     " <dedispersionStepSize value=\"0.1\" />"
                     " <numberOfBuffers value=\"3\" />"
                     "</DedispersionModule>"
                 "</modules>")
                  .arg( QString(TEST_DATA_DIR) + QDir::separator() + "band31.bp")
                  .arg( 100 )
                  .arg( 150 )
                  .arg( 0.1 )
                  .arg( -0.2 )
                  .arg ( 200 );
     try {
         DedispersionPipeline p(streamId);
         LofarPipelineTester tester(&p, config(xml));
         tester.run();
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

} // namespace lofar
} // namespace pelican
