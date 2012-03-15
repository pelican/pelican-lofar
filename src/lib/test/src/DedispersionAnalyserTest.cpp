#include "DedispersionAnalyserTest.h"
#include "DedispersionAnalyser.h"
#include "DedispersionDataAnalysis.h"
#include "DedispersionDataGenerator.h"
#include "DedispersionSpectra.h"


namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( DedispersionAnalyserTest );
/**
 *@details DedispersionAnalyserTest 
 */
DedispersionAnalyserTest::DedispersionAnalyserTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
DedispersionAnalyserTest::~DedispersionAnalyserTest()
{
}

void DedispersionAnalyserTest::setUp()
{
}

void DedispersionAnalyserTest::tearDown()
{
}

void DedispersionAnalyserTest::test_noSignificantEvents()
{
     // Use Case:
     // No single significant dedispersion event
     // Expect:
     // No Events reported
     DedispersionSpectra inputData;
     DedispersionDataAnalysis outputData;

     ConfigNode config;
     DedispersionAnalyser analyser(config);
     CPPUNIT_ASSERT_EQUAL( 0, analyser.analyse( &inputData, &outputData ) );
}

void DedispersionAnalyserTest::test_singleEvent()
{
     // Use Case:
     // Single significant dedispersion event
     // Integration test with Dedispersion Module
     // Expect:
     // Event to be reported
     try {
         DedispersionDataGenerator dataGenerator;
         DedispersionSpectra* inputData = dataGenerator.dedispersionData(10);

         // -- Setup Data
         DedispersionDataAnalysis outputData;

         ConfigNode config;
         DedispersionAnalyser analyser(config);
         CPPUNIT_ASSERT_EQUAL( 1, analyser.analyse( inputData, &outputData ) );
         DedispersionEvent e = outputData.events()[0];

         // clean up
         dataGenerator.deleteData(inputData);
     } catch ( const QString& e ) {
        CPPUNIT_FAIL( "THROW: " + e.toStdString() );
     }
}

} // namespace lofar
} // namespace pelican
