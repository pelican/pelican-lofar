#include "DedispersionPipelineTest.h"
#include "DedispersionPipeline.h"
#include "test/LofarPipelineTester.h"


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
     QString streamId = "LofarDataStream1";
     QString xml="<pipelineConfig />";
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
