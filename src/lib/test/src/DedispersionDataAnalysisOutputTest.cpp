#include "DedispersionDataAnalysisOutputTest.h"
#include "DedispersionDataAnalysisOutput.h"
#include "DedispersionDataAnalysis.h"
#include "DedispersionSpectra.h"
#include "SpectrumDataSet.h"
#include "pelican/utility/ConfigNode.h"
#include "pelican/utility/test/TestFile.h"
#include <QFile>
#include <QDebug>


namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( DedispersionDataAnalysisOutputTest );
/**
 *@details DedispersionDataAnalysisOutputTest 
 */
DedispersionDataAnalysisOutputTest::DedispersionDataAnalysisOutputTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
DedispersionDataAnalysisOutputTest::~DedispersionDataAnalysisOutputTest()
{
}

void DedispersionDataAnalysisOutputTest::setUp()
{
}

void DedispersionDataAnalysisOutputTest::tearDown()
{
}

void DedispersionDataAnalysisOutputTest::test_method()
{
    // create an ouput stream
    test::TestFile file(true);
    QString filename = file.filename();
    SpectrumDataSetStokes sblob;
    QList<SpectrumDataSetStokes*> sblobs;
    sblobs.append( &sblob );
    DedispersionSpectra data; data.resize(10,10,1.0,0.1);
    data.setInputDataBlobs( sblobs );
    DedispersionDataAnalysis blob1;
    blob1.reset(&data);
    blob1.addEvent( 1, 1);
    CPPUNIT_ASSERT_EQUAL( 1, blob1.eventsFound() );
    ConfigNode config;
    {
        DedispersionDataAnalysisOutput writer(config);
        writer.addFile(filename);
        writer.send("stream1", &blob1);
    }
    QFile f(filename);
    CPPUNIT_ASSERT( f.exists(filename) );
    CPPUNIT_ASSERT( f.size() > 0 );
}

} // namespace lofar
} // namespace pelican
