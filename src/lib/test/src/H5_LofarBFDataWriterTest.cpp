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

// Test class to enable provide a non-op version for testing
class TestH5_LofarBFDataWriter : public H5_LofarBFDataWriter {
    public:
    TestH5_LofarBFDataWriter( const ConfigNode& config ) : H5_LofarBFDataWriter(config) { _stokesType = STOKES_I; };
    ~TestH5_LofarBFDataWriter() {};
    virtual void _writeData( const SpectrumDataSetBase* ) {};
};

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
    _fileDir = "/_H5_LofarBFDataWriterTest_";
#if QT_VERSION >= 0x040400
    _fileDir += QString().setNum( QCoreApplication::applicationPid() );
#endif
    _fullFileDir=QDir::tempPath() + _fileDir;
    if( ! QDir::temp().mkpath( _fullFileDir ) )
        CPPUNIT_FAIL("unable to create temporary directory " + _fullFileDir.toStdString() );
}

void H5_LofarBFDataWriterTest::tearDown()
{
    // attempt to clean up the temp directory
    QDir dir( QDir::tempPath() + "/" + _fileDir ); // dont use _fullFilePath for safety
    if (dir.exists()) {
        foreach(QFileInfo info, dir.entryInfoList(QDir::NoDotAndDotDot | QDir::Files )) {
            QFile::remove(info.absoluteFilePath());
        }
        //std::cout << "remove dir " << _fullFileDir.toStdString();
        dir.rmdir( QDir::tempPath() + _fileDir);
    }
}

void H5_LofarBFDataWriterTest::test_method()
{
    // generate some test data
    DedispersionDataGenerator stokesData;
    unsigned nSamples = 20;
    unsigned nBlocks = 64;
    unsigned nSubbands = 4;
    float dm = 10.0;
    unsigned pol=0; // only one polarisation
    stokesData.setTimeSamplesPerBlock( nSamples );
    stokesData.setSubbands( nSubbands );
    QList<SpectrumDataSetStokes*> spectrumData = stokesData.generate( nBlocks, dm );
    stokesData.setSubbands( nSubbands* 2 );
    QList<SpectrumDataSetStokes*> spectrumData2 = stokesData.generate( nBlocks *2, dm );

    try {
    { // Use Case:
      // No Configuration given
      // Expect:
      // Be able to construct the object
      QString xml;
      ConfigNode c;
      TestH5_LofarBFDataWriter out( c );
    }
    { // Use Case:
      // Empty Configuration
      // Single DataBlob
      // Expect:
      // Files to be generated
      QString xml = "<H5_LofarBFDataWriter>\n" +
                    QString("<file filepath=\"%1\"").arg( _fullFileDir )
                    + " />"
                    "</H5_LofarBFDataWriter>";
      ConfigNode c;
      c.setFromString(xml);
      QString rawFile, h5File;

      CPPUNIT_ASSERT_EQUAL( 1, (int)spectrumData[0]->nPolarisations() );
      TestH5_LofarBFDataWriter out( c );
      out.send("data", spectrumData[0] );
      rawFile = out.rawFilename( pol );
      h5File = out.metaFilename( pol );
      // check files exists
      QFile f(rawFile);
      CPPUNIT_ASSERT( f.exists() );
      QFile hf(h5File);
      CPPUNIT_ASSERT( hf.exists() );

      // add more data of the same dimension 
      // epect the raw data file to remain 0 size as this is
      // not implemented in the base class
      // The current file names should be the same
      out.send("data", spectrumData[1] );
      CPPUNIT_ASSERT_EQUAL( rawFile.toStdString(), out.rawFilename( pol ).toStdString() );
      CPPUNIT_ASSERT_EQUAL( h5File.toStdString(), out.metaFilename( pol ).toStdString() );

      // add more data of different dimension
      // expect new files to be generated
      out.send("data", spectrumData2[0] );
      QString rawFile2 = out.rawFilename( pol );
      CPPUNIT_ASSERT( rawFile != rawFile2 );
      CPPUNIT_ASSERT( h5File != out.metaFilename( pol ) );
      QFile f2(rawFile2);
      CPPUNIT_ASSERT( f2.exists() );

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
