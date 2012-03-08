#include "DedispersionModuleTest.h"
#include "DedispersionModule.h"
#include "pelican/utility/ConfigNode.h"
#include "SpectrumDataSet.h"
#include "WeightedSpectrumDataSet.h"
#include "DedispersionDataGenerator.h"
#include <boost/bind.hpp>
#include <iostream>
#include "pelican/utility/ConfigNode.h"
#include <QDebug>


namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( DedispersionModuleTest );
/**
 *@details DedispersionModuleTest 
 */
DedispersionModuleTest::DedispersionModuleTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
DedispersionModuleTest::~DedispersionModuleTest()
{
}

void DedispersionModuleTest::setUp()
{
}

void DedispersionModuleTest::tearDown()
{
}

void DedispersionModuleTest::test_multipleBlobs ()
{
     // Use Case:
     // Dedispersed signal spread over multiple data blobs
     // Data per blob is the same size as the databuffer
     // Expect:
     // two calls to the gpu.
     // Overlap of data from the first blob inline with the maxshift 
     // parameter
     // In time the blobs will overlap thus:
     // | data1      |
     //          |  data2     |
     // i.e. the tail end of datablob1 will be duplicated to the beginning of
     // the second buffer.
     // As the buffer is of fixed size, all the data in the second blob will
     // not be included, but should be placed in the next awaiting buffer
     try {
        float dm = 10.0;
        unsigned ddSamples = 200;
        unsigned nBlocks = 2;
        unsigned nSamples = 6400;
        DedispersionDataGenerator stokesData;
        stokesData.setTimeSamplesPerBlock( nSamples );
        QList<SpectrumDataSetStokes*> spectrumData = stokesData.generate( nBlocks, dm );
        stokesData.writeToFile( "inputStokes.data", spectrumData );

        WeightedSpectrumDataSet weightedData(spectrumData[0]);
        WeightedSpectrumDataSet weightedData2(spectrumData[1]);
        ConfigNode config;
        CPPUNIT_ASSERT_EQUAL( nSamples, spectrumData[0]->nTimeBlocks() );
        // setup configuration
        QString configString = QString("<DedispersionModule>"
                " <sampleNumber value=\"%1\" />"
                " <frequencyChannel1 value=\"%2\"/>"
                " <sampleTime value=\"%3\"/>"
                " <channelBandwidth value=\"%4\"/>"
                " <dedispersionSamples value=\"%5\" />"
                " <dedispersionStepSize value=\"0.1\" />"
                " <numberOfBuffers value=\"3\" />"
                "</DedispersionModule>")
                .arg( nSamples ) // block size should match the buffer size to ensure we get two calls to the GPU
                .arg( stokesData.startFrequency())
                .arg( stokesData.timeOfSample())
                .arg( stokesData.bandwidthOfSample())
                .arg( ddSamples );
          config.setFromString(configString);

          DedispersionModule ddm(config);
          LockingPtrContainer<DedispersionSpectra* >* buffer = outputBuffer(2);
          ddm.connect( boost::bind( &DedispersionModuleTest::connected, this, _1 ) );
          _connectData = 0;
          _connectCount = 0;
          ddm.dedisperse( &weightedData, buffer ); // asynchronous task
          CPPUNIT_ASSERT_EQUAL( 1, buffer->numberAvailable() );
          while( _connectCount != 1 ) { sleep(1); };
          float expectedDMIntentsity = spectrumData[0]->nSubbands() * spectrumData[0]->nChannels();
          CPPUNIT_ASSERT_EQUAL( expectedDMIntentsity , _connectData->dm( 0, dm ) );
          ddm.dedisperse( &weightedData2, buffer ); // asynchronous task
          while( _connectCount != 2 ) { sleep(1); };
          destroyBuffer( buffer );
          stokesData.deleteData(spectrumData);
     }
    catch( const QString& s )
    {
        CPPUNIT_FAIL(s.toStdString());
    }
}

void DedispersionModuleTest::test_method()
{
    try {
        float dm = 10.0;
        unsigned ddSamples = 200;
        unsigned nBlocks = 1;
        unsigned nSamples = 6400;
        DedispersionDataGenerator stokesData;
        stokesData.setTimeSamplesPerBlock( nSamples );
        QList<SpectrumDataSetStokes*> spectrumData = stokesData.generate( nBlocks, dm );
        //stokesData.writeToFile( "inputStokes.data", spectrumData );
        
        WeightedSpectrumDataSet weightedData(spectrumData[0]);
        { // Use Case:
          // Single Data Blob as input, of same size as the buffer
          // Expect:
          // output dataBlob to be filled
          // returned output blob should be the same as that passed
          // to any connected functions
          ConfigNode config;
          CPPUNIT_ASSERT_EQUAL( nSamples, spectrumData[0]->nTimeBlocks() );
          QString configString = QString("<DedispersionModule>"
                                         " <sampleNumber value=\"%1\" />"
                                         " <frequencyChannel1 value=\"%2\"/>"
                                         " <sampleTime value=\"%3\"/>"
                                         " <channelBandwidth value=\"%4\"/>"
                                         " <dedispersionSamples value=\"%5\" />"
                                         " <dedispersionStepSize value=\"0.1\" />"
                                         "</DedispersionModule>")
                                        .arg( nSamples )
                                        .arg( stokesData.startFrequency())
                                        .arg( stokesData.timeOfSample())
                                        .arg( stokesData.bandwidthOfSample())
                                        .arg( ddSamples );
          config.setFromString(configString);
          DedispersionModule ddm(config);
          LockingPtrContainer<DedispersionSpectra* >* buffer = outputBuffer(2);
          ddm.connect( boost::bind( &DedispersionModuleTest::connected, this, _1 ) );
          _connectData = 0;
          _connectCount = 0;
          ddm.dedisperse( &weightedData, buffer ); // asynchronous task
          while( ! _connectCount ) { sleep(1); };
          CPPUNIT_ASSERT_EQUAL( 1, _connectCount );
          int outputSampleSize = (int)(((nSamples - ddm.maxshift() )));
          CPPUNIT_ASSERT_EQUAL( (int)(outputSampleSize*ddSamples), _connectData->data().size() );
// Print out the resulting data
//          std::ofstream file("output.data");
//          for( int i=0; i < buffer->current()->data().size(); ++i ) {
//                file << (buffer->current()->data())[i] << std::string(((i+1)%outputSampleSize)?" ":"\n");
//          }
//          std::cout << std::endl;

          float expectedDMIntentsity = spectrumData[0]->nSubbands() * spectrumData[0]->nChannels();
          CPPUNIT_ASSERT_EQUAL( expectedDMIntentsity , _connectData->dm( 0, dm ) );
          destroyBuffer( buffer );
          stokesData.deleteData(spectrumData);
        }
    }
    catch( QString s )
    {
        CPPUNIT_FAIL(s.toStdString());
    }
}

void DedispersionModuleTest::connected( DataBlob* dataOut ) {
    ++_connectCount;
    _connectData = 0;
    CPPUNIT_ASSERT( ( _connectData = dynamic_cast<DedispersionSpectra* >(dataOut) ) );
    // check the dedispersion values
}

ConfigNode DedispersionModuleTest::testConfig(QString xml) const
{
    ConfigNode node;
    if( xml == "" ) {
        xml = QString("<DedispersionModule >\n"
                  "<Band startFrequency=\"131.250763\" endFrequency=\"137.3\" />\n"
              "</DedispersionModule>\n");
    }
    node.setFromString( xml );
    return node;
}

LockingPtrContainer<DedispersionSpectra* >* DedispersionModuleTest::outputBuffer(int size) {
     QList<DedispersionSpectra* >* buffer = new QList< DedispersionSpectra* >;
     for(int i=0; i < size; ++i ) {
        buffer->append( new DedispersionSpectra );
     }
     return new LockingPtrContainer<DedispersionSpectra* >( buffer );
}

void DedispersionModuleTest::destroyBuffer(
        LockingPtrContainer<DedispersionSpectra* >* b) 
{
    foreach( DedispersionSpectra* d, *(b->rawBuffer()) ) {
        delete d;
    }
    delete b->rawBuffer();
}

} // namespace lofar
} // namespace pelican
