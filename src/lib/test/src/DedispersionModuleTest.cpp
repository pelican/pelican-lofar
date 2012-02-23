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
        stokesData.writeToFile( "inputStokes.data", spectrumData );
        
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
          LockingCircularBuffer<DedispersionSpectra* >* buffer = outputBuffer(2);
          ddm.connect( boost::bind( &DedispersionModuleTest::connected, this, _1 ) );
          DedispersionSpectra* data = ddm.dedisperse( &weightedData, buffer ); // asynchronous task
          _connectData = 0;
          _connectCount = 0;
          while( ! _connectCount ) { sleep(1); };
          CPPUNIT_ASSERT_EQUAL( 1, _connectCount );
          CPPUNIT_ASSERT_EQUAL( data, _connectData );
          int outputSampleSize = (int)(((nSamples - ddm.maxshift() )));
          CPPUNIT_ASSERT_EQUAL( (int)(outputSampleSize*ddSamples), buffer->current()->data().size() );
          std::ofstream file("output.data");
          //foreach( float d, buffer->current()->data() ) {
          for( int i=0; i < buffer->current()->data().size(); ++i ) {
                file << (buffer->current()->data())[i] << std::string(((i+1)%outputSampleSize)?" ":"\n");
          }
          std::cout << std::endl;

          float expectedDMIntentsity = spectrumData[0]->nSubbands() * spectrumData[0]->nChannels();
          CPPUNIT_ASSERT_EQUAL( expectedDMIntentsity , buffer->current()->dm( 0, dm ) );
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

LockingCircularBuffer<DedispersionSpectra* >* DedispersionModuleTest::outputBuffer(int size) {
     QList<DedispersionSpectra* >* buffer = new QList< DedispersionSpectra* >;
     for(int i=0; i < size; ++i ) {
        buffer->append( new DedispersionSpectra );
     }
     return new LockingCircularBuffer<DedispersionSpectra* >( buffer );
}

void DedispersionModuleTest::destroyBuffer(
        LockingCircularBuffer<DedispersionSpectra* >* b) 
{
    foreach( DedispersionSpectra* d, *(b->rawBuffer()) ) {
        delete d;
    }
}

} // namespace lofar
} // namespace pelican
