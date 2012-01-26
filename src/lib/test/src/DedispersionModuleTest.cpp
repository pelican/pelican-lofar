#include "DedispersionModuleTest.h"
#include "DedispersionModule.h"
#include "pelican/utility/ConfigNode.h"
#include "SpectrumDataSet.h"
#include "WeightedSpectrumDataSet.h"
#include <boost/bind.hpp>
#include <iostream>


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
    unsigned nSamples = 10;
    unsigned nChannels = 10;
    unsigned nPolarisations=2;
    unsigned nSubbands=2;
    try {
        SpectrumDataSet<float> spectrumData;
        spectrumData.resize( nSamples, nSubbands, nPolarisations, nChannels );
        WeightedSpectrumDataSet weightedData(&spectrumData);
        { // Use Case:
          // Single Data Blob as input, of same size as the buffer
          // Expect:
          // output dataBlob to be filled
          // returned output blob should be the same as that passed
          // to any connected functions
          ConfigNode config;
          QString configString = QString("<DedispersionModule><dedispersionSampleNumber value=\"%1\" /><frequencyChannel1 value=\"100\"/></DedispersionModule>").arg(nSamples);
          config.setFromString(configString);
          DedispersionModule dm(config);
          LockingCircularBuffer<DedispersedTimeSeries<float>* >* buffer = outputBuffer(2);
          dm.connect( boost::bind( &DedispersionModuleTest::connected, this, _1 ) );
          DedispersedTimeSeries<float>* data = dm.dedisperse( &weightedData, buffer ); // asynchronous task
          _connectData = 0;
          _connectCount = 0;
          while( ! _connectCount ) { sleep(1); };
          CPPUNIT_ASSERT_EQUAL( 1, _connectCount );
          CPPUNIT_ASSERT_EQUAL( data, _connectData );
          destroyBuffer( buffer );
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
std::cout << "connected" << std::endl;
    CPPUNIT_ASSERT( ( _connectData = dynamic_cast<DedispersedTimeSeries<float>* >(dataOut) ) );
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

LockingCircularBuffer<DedispersedTimeSeries<float>* >* DedispersionModuleTest::outputBuffer(int size) {
     QList<DedispersedTimeSeries<float>* >* buffer = new QList< DedispersedTimeSeries<float>* >;
     for(int i=0; i < size; ++i ) {
        buffer->append( new DedispersedTimeSeries<float> );
     }
     return new LockingCircularBuffer<DedispersedTimeSeries<float>* >( buffer );
}

void DedispersionModuleTest::destroyBuffer(
        LockingCircularBuffer<DedispersedTimeSeries<float>* >* b) 
{
    foreach( DedispersedTimeSeries<float>* d, *(b->rawBuffer()) ) {
        delete d;
    }
}

void DedispersionModuleTest::_deleteStokesData( QList<SpectrumDataSetStokes*>& data ) {
    foreach( SpectrumDataSetStokes* d, data ) {
        delete d;
    }
}

QList<SpectrumDataSetStokes*> DedispersionModuleTest::_generateStokesData(int numberOfBlocks, float dm ) {
    unsigned nSamples = 16; // samples per blob
    unsigned nSubbands = 32;
    unsigned nChannels = 64; // 2048 total channels (32x64)

    double fch1 = 150;
    double foff = -6.0/2048.0;
    double tsamp = 327.68; // time sample length
    QList<SpectrumDataSetStokes*> data;

    for( int i=0; i < numberOfBlocks; ++i ) {
        SpectrumDataSetStokes* stokes = new SpectrumDataSetStokes;
        stokes->resize(nSamples, nSubbands, 1, nChannels);
        data.append(stokes);

        int offset = (i - 1) * nSamples;
        //stokes->setLofarTimestamp(channeliserOutput->getLofarTimestamp());
        for (unsigned int t = 0; t < nSamples; ++t ) {
            for (unsigned s = 0; s < nSubbands; ++s ) {
                for (unsigned c = 0; c < nChannels; ++c) {
                    int absChannel = nSubbands * nChannels + nChannels;
                    int index = (int)(dm * (4148.741601 * ((1.0 / (fch1 + (foff * absChannel)) /
                                    (fch1 + (foff * absChannel))) - (1.0 / fch1 / fch1))) /tsamp);
                    int sampleNumber = index - offset;
                    float* I = stokes->spectrumData(t, s, 0);
                    if( sampleNumber == (int)t ) {
                        I[c] = 1.0;
                    } else {
                        I[c] = 0.0;
                    }
                }
            }
        }
    }
    return data;
}

} // namespace lofar
} // namespace pelican
