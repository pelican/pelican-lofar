#include "H5_LofarBFDataWriterTest.h"
#include "H5_LofarBFDataWriter.h"
#include <QDir>
#include <QFile>
#include <QCoreApplication>
#include <QFileInfo>
#include "DedispersionDataGenerator.h"
#include "SpectrumDataSet.h"


namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( H5_LofarBFDataWriterTest );
/**
 *@details H5_LofarBFDataWriterTest 
 */
H5_LofarBFDataWriterTest::H5_LofarBFDataWriterTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
H5_LofarBFDataWriterTest::~H5_LofarBFDataWriterTest()
{
}

void H5_LofarBFDataWriterTest::setUp()
{
    _fileDir = QDir::tempPath() + "/_H5_LofarBFDataWriterTest_";
#if QT_VERSION >= 0x040400
    _fileDir += QString().setNum( QCoreApplication::applicationPid() );
#endif

}

void H5_LofarBFDataWriterTest::tearDown()
{
     // attempt to clean up the temp directory
     QDir dir(_fileDir);
     if( dir.match( QDir::tempPath()+"/*", _fileDir ) ) { // avoid any nasty surprises
         if (dir.exists()) {
            foreach(QFileInfo info, dir.entryInfoList(QDir::NoDotAndDotDot | QDir::Files )) {
      //          QFile::remove(info.absoluteFilePath());
std::cout << "remove file " << info.absoluteFilePath().toStdString();

            }
std::cout << "remove dir " << _fileDir.toStdString();
      //      dir.rmdir(_fileDir);
        }
     }
}

void H5_LofarBFDataWriterTest::test_method()
{
    // generate some test data
    DedispersionDataGenerator stokesData;
    unsigned nSamples = 200;
    unsigned nBlocks = 64;
    float dm = 10.0;
    stokesData.setTimeSamplesPerBlock( nSamples );
    QList<SpectrumDataSetStokes*> spectrumData = stokesData.generate( nBlocks, dm );
    QList<SpectrumDataSetStokes*> spectrumData2 = stokesData.generate( nBlocks *2, dm );

    try {
    { // Use Case:
      // No Configuration given
      // Expect:
      // Be able to construct the object
      QString xml;
      ConfigNode c;
      H5_LofarBFDataWriter out( c );
    }
    { // Use Case:
      // Empty Configuration
      // Single DataBlob
      // Expect:
      // Files to be generated
      QString xml = "<H5_LofarBFDataWriter>\n"
                    "</H5_LofarBFDataWriter>";
      ConfigNode c;
      c.setFromString(xml);
      QString rawFile, h5File;

      H5_LofarBFDataWriter out( c );
      out.send("data", spectrumData[0] );
      rawFile = out.rawFilename();
      h5File = out.metaFilename();
      // check files exists
      QFile f(rawFile);
      CPPUNIT_ASSERT( f.exists() );
      CPPUNIT_ASSERT_EQUAL( spectrumData[0]->size(), (int)f.size() );
      QFile hf(h5File);
      CPPUNIT_ASSERT( hf.exists() );

      // add more data of the same dimension 
      // expect the raw data file to increase in size
      // and the current file names to be the same
      out.send("data", spectrumData[1] );
      CPPUNIT_ASSERT_EQUAL( rawFile.toStdString(), out.rawFilename().toStdString() );
      CPPUNIT_ASSERT_EQUAL( h5File.toStdString(), out.metaFilename().toStdString() );
      CPPUNIT_ASSERT_EQUAL( spectrumData[0]->size() + spectrumData[1]->size(), (int)f.size() );

      // add more data of different dimension
      // expect new files to be generated
      QString rawFile2 = out.rawFilename();
      out.send("data", spectrumData2[0] );
      CPPUNIT_ASSERT( rawFile != rawFile2 );
      CPPUNIT_ASSERT( h5File != out.metaFilename() );
      QFile f2(rawFile2);
      CPPUNIT_ASSERT( f2.exists() );
      CPPUNIT_ASSERT_EQUAL( spectrumData[2]->size(), (int)f2.size() );

    }
    } catch( QString& s )
    {
        CPPUNIT_FAIL(s.toStdString());
    }
    stokesData.deleteData(spectrumData2);
    stokesData.deleteData(spectrumData);
}

} // namespace lofar
} // namespace pelican
