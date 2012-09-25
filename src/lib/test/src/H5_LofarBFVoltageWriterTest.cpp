#include "H5_LofarBFVoltageWriterTest.h"
#include "H5_LofarBFVoltageWriter.h"
#include "SpectrumDataSet.h"
#include "SpectrumDataSet.h"
#include "TestDir.h"
#include "pelican/utility/ConfigNode.h"
#include "TimerData.h"


namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( H5_LofarBFVoltageWriterTest );
/**
 *@details H5_LofarBFVoltageWriterTest 
 */
H5_LofarBFVoltageWriterTest::H5_LofarBFVoltageWriterTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
H5_LofarBFVoltageWriterTest::~H5_LofarBFVoltageWriterTest()
{
}

void H5_LofarBFVoltageWriterTest::setUp()
{
    _testDir = new test::TestDir( "H5_LofarBFVoltageWriter", true );
}

void H5_LofarBFVoltageWriterTest::tearDown()
{
    delete _testDir;
}

void H5_LofarBFVoltageWriterTest::test_method()
{
    int polMax=4; // all 4 components
    SpectrumDataSetC32 data;
    data.resize(10,10,2,10);
    
    try {
        // Use case:
        // Stream out - most checks are done in the Base class test
        // so we only need to check that the output file is of the
        // correct size
      QString xml = "<H5_LofarBFVoltageWriter>\n" +
                   QString("<file filepath=\"%1\"").arg( _testDir->absolutePath() )
                    + " />"
                    "<checkPoint interval=\"1\" />"
                    "</H5_LofarBFVoltageWriter>";
      ConfigNode c;
      c.setFromString(xml);
      QString rawFile, h5File;
      H5_LofarBFVoltageWriter out( c );
      out.send("data", &data );
      out.flush();
      for(int pol=0; pol < polMax; ++pol ) {
          rawFile = out.rawFilename( pol );
          h5File = out.metaFilename( pol );
          QFile f(rawFile);
          CPPUNIT_ASSERT( f.exists() );
          QFile hf(h5File);
          CPPUNIT_ASSERT( hf.exists() );
          int polarisationSetSize=(int)(data.size()/polMax * 
                                        sizeof(std::complex<float>));
          CPPUNIT_ASSERT_EQUAL( polarisationSetSize, (int)f.size() ); 
          // add more data to verify file grows
          //out.send("data", &data );
          //CPPUNIT_ASSERT_EQUAL( 2 * polarisationSetSize , (int)f.size() );
      }

    } catch( QString& s )
    {
        CPPUNIT_FAIL(s.toStdString());
    }
}

void H5_LofarBFVoltageWriterTest::test_performance()
{
    SpectrumDataSetC32 data;
    data.resize(128*16,31,2,64); // timeblocks, subbands, pol, channels
    
    try {
        // Use case:
        // Stream out - most checks are done in the Base class test
        // so we only need to check that the output file is of the
        // correct size
      QString xml = "<H5_LofarBFVoltageWriter>\n" +
                   QString("<file filepath=\"%1\"").arg( _testDir->absolutePath() )
                    + " />"
                    "<checkPoint interval=\"100000\" />"
                    "</H5_LofarBFVoltageWriter>";
      ConfigNode c;
      c.setFromString(xml);
      H5_LofarBFVoltageWriter out( c );
      TimerData t;
      int sampleIterations = 10;
      int sampleSize = data.size() * sampleIterations * sizeof(std::complex<float>) * 8;
      for( int j=0; j<15; ++j ) {
          t.tick();
          for(int i=0; i<sampleIterations; ++i ) {
              out.send("data", &data );
          }
          t.tock();
          float rate = (sampleSize/t.timeElapsed)/1e6;
          std::cout << "Data Write rate = "  << rate << " MBits/s" << std::endl;
      }
    } catch( QString& s )
    {
        CPPUNIT_FAIL(s.toStdString());
    }
}

} // namespace lofar
} // namespace pelican
