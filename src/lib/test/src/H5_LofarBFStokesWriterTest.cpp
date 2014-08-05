#include "H5_LofarBFStokesWriterTest.h"
#include "H5_LofarBFStokesWriter.h"
#include "DedispersionDataGenerator.h"
#include "SpectrumDataSet.h"
#include "TestDir.h"
#include "pelican/utility/ConfigNode.h"


namespace pelican {

namespace ampp {

CPPUNIT_TEST_SUITE_REGISTRATION( H5_LofarBFStokesWriterTest );
/**
 *@details H5_LofarBFStokesWriterTest 
 */
H5_LofarBFStokesWriterTest::H5_LofarBFStokesWriterTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
H5_LofarBFStokesWriterTest::~H5_LofarBFStokesWriterTest() {
}

void H5_LofarBFStokesWriterTest::setUp()
{
    _testDir = new test::TestDir( "H5_LofarBFStokesWriter", true );
}

void H5_LofarBFStokesWriterTest::tearDown()
{
    delete _testDir;
}

void H5_LofarBFStokesWriterTest::test_method()
{
    // generate test data
    DedispersionDataGenerator stokesData;
    unsigned nSamples = 20;
    unsigned nBlocks = 64;
    unsigned nSubbands = 4;
    float dm = 10.0;
    unsigned pol=0; // only one polarisation
    stokesData.setTimeSamplesPerBlock( nSamples );
    stokesData.setSubbands( nSubbands );
    QList<SpectrumDataSetStokes*> spectrumData = stokesData.generate( nBlocks, dm );
    try {
        // Use case:
        // Stream out - most checks are done in the Base class test
        // so we only need to check that the output file is of the
        // correct size
      QString xml = "<H5_LofarBFStokesWriter>\n" +
                   QString("<file filepath=\"%1\"").arg( _testDir->absolutePath() )
                    + " />"
                    "<checkPoint interval=\"1\" />"
                    "</H5_LofarBFStokesWriter>";
      ConfigNode c;
      c.setFromString(xml);
      QString rawFile, h5File;
      H5_LofarBFStokesWriter out( c );
      out.send("data", spectrumData[0] );
      out.flush();
      rawFile = out.rawFilename( pol );
      h5File = out.metaFilename( pol );
      QFile f(rawFile);
      CPPUNIT_ASSERT( f.exists() );
      QFile hf(h5File);
      CPPUNIT_ASSERT( hf.exists() );
      CPPUNIT_ASSERT_EQUAL( (int)(spectrumData[0]->size() * sizeof(float)), (int)f.size() ); 
      // add more data to verify file grows
      out.send("data", spectrumData[1] );
      out.flush();
      CPPUNIT_ASSERT_EQUAL( (int)(spectrumData[0]->size() + spectrumData[1]->size()) * (int)sizeof(float) , (int)f.size() );

    } catch( QString& s )
    {
        CPPUNIT_FAIL(s.toStdString());
    }
    stokesData.deleteData(spectrumData);
}

} // namespace ampp
} // namespace pelican
