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
    try {
        float dm = 35.5;
        QList<SpectrumDataSetStokes*> spectrumData = _generateStokesData( 1, dm );
        WeightedSpectrumDataSet weightedData(spectrumData[0]);
        { // Use Case:
          // Single Data Blob as input, of same size as the buffer
          // Expect:
          // output dataBlob to be filled
          // returned output blob should be the same as that passed
          // to any connected functions
          ConfigNode config;
          unsigned nSamples=spectrumData[0]->nTimeBlocks();
          QString configString = QString("<DedispersionModule>"
                                         " <dedispersionSampleNumber value=\"%1\" />"
                                         " <frequencyChannel1 value=\"150\"/>"
                                         " <channelBandwidth value=\"-0.00292969\"/>"
                                         " <sampleTime value=\"327.68\"/>"
                                         "</DedispersionModule>").arg(nSamples);
          config.setFromString(configString);
          DedispersionModule dm(config);
          LockingCircularBuffer<DedispersionSpectra* >* buffer = outputBuffer(2);
          dm.connect( boost::bind( &DedispersionModuleTest::connected, this, _1 ) );
          DedispersionSpectra* data = dm.dedisperse( &weightedData, buffer ); // asynchronous task
          _connectData = 0;
          _connectCount = 0;
          while( ! _connectCount ) { sleep(1); };
          CPPUNIT_ASSERT_EQUAL( 1, _connectCount );
          CPPUNIT_ASSERT_EQUAL( data, _connectData );
          destroyBuffer( buffer );
          _deleteStokesData(spectrumData);
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
    double tsamp = 0.00032768; // time sample length
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
                    int absChannel = s * nChannels + c;
                    int index = (int)( (4148.741601 * ((1.0 / (fch1 + (foff * absChannel)) /
                                    (fch1 + (foff * absChannel))) - (1.0 / fch1 / fch1)))/tsamp );
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
