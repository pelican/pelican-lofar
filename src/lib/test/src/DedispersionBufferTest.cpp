#include "DedispersionBufferTest.h"
#include "DedispersionBuffer.h"
#include "SpectrumDataSet.h"
#include "WeightedSpectrumDataSet.h"


namespace pelican {

namespace ampp {

CPPUNIT_TEST_SUITE_REGISTRATION( DedispersionBufferTest );
/**
 *@details DedispersionBufferTest 
 */
DedispersionBufferTest::DedispersionBufferTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
DedispersionBufferTest::~DedispersionBufferTest()
{
}

void DedispersionBufferTest::setUp()
{
}

void DedispersionBufferTest::tearDown()
{
}

void DedispersionBufferTest::test_sizing()
{
     unsigned sampleNumber = 0;
     unsigned nSamples = 10;
     unsigned nChannels = 10;
     unsigned nPolarisations=2;
     unsigned nSubbands=2;
     unsigned sampleSize = nSubbands * nPolarisations * nChannels;
     WeightedSpectrumDataSet data;
     SpectrumDataSetStokes sdata;
     QVector<float> noise;
     //     SpectrumDataSetStokes data;
     sdata.resize( nSamples, nSubbands, nPolarisations, nChannels );
     { // Use Case:
       // Default constructor
       // Expect:
       // Zero size
       // addSamples to do nothing

       DedispersionBuffer b;
       CPPUNIT_ASSERT_EQUAL(0, (int)b.size() );
       CPPUNIT_ASSERT_EQUAL((unsigned int)0, b.numSamples() );
       CPPUNIT_ASSERT_EQUAL((unsigned int)0, b.spaceRemaining() );
       CPPUNIT_ASSERT_EQUAL( b.maxSamples() , b.addSamples( &data, noise, &sampleNumber ));
       CPPUNIT_ASSERT_EQUAL( (unsigned)0, sampleNumber );
     }
     { // Use Case:
       // constructor with fixed size of samples, sized correctly
       // addSamples supplied with samples to exactly fill the buffer
       // Expect:
       // size as set, addSamples to complete OK, with 0 space left
       DedispersionBuffer b(nSamples,sampleSize );
       CPPUNIT_ASSERT_EQUAL( (unsigned)0, sampleNumber );
       CPPUNIT_ASSERT_EQUAL(nSamples * sampleSize* sizeof(float), b.size() );
       CPPUNIT_ASSERT_EQUAL((unsigned int)0, b.numSamples() );
       CPPUNIT_ASSERT_EQUAL(nSamples, b.spaceRemaining() );
       CPPUNIT_ASSERT_EQUAL( (unsigned)0 , b.addSamples( &data, noise, &sampleNumber ));
       CPPUNIT_ASSERT_EQUAL( nSamples, b.numSamples() );
       CPPUNIT_ASSERT_EQUAL((unsigned int)0, b.spaceRemaining() );
       CPPUNIT_ASSERT_EQUAL( nSamples, sampleNumber );
       // we should not be able to add any more data as the buffer is full
       sampleNumber = 0;
       CPPUNIT_ASSERT_EQUAL( (unsigned)0 , b.addSamples( &data, noise, &sampleNumber ));
       CPPUNIT_ASSERT_EQUAL( (unsigned)0, sampleNumber );
       // clear it and we should be able to use it again
       b.clear();
       CPPUNIT_ASSERT_EQUAL((unsigned int)0, b.numSamples() );
       CPPUNIT_ASSERT_EQUAL((unsigned int)10, b.spaceRemaining() );
       CPPUNIT_ASSERT_EQUAL( (unsigned)0 , b.addSamples( &data, noise, &sampleNumber ));
       CPPUNIT_ASSERT_EQUAL( nSamples, b.numSamples() );
       CPPUNIT_ASSERT_EQUAL((unsigned int)0, b.spaceRemaining() );
     }
     { // Use Case:
       // constructor with fixed size of samples, sized correctly
       // call addSamples with data that will overflow the buffer 
       // Expect:
       // addSamples to complete OK, with no space left in the buffer and sample number set correctly
       unsigned samples = nSamples -2;
       DedispersionBuffer b( samples, sampleSize );
       sampleNumber = 0;
       CPPUNIT_ASSERT_EQUAL( (unsigned)0 , b.addSamples( &data, noise, &sampleNumber ));
       CPPUNIT_ASSERT_EQUAL((unsigned int)0, b.spaceRemaining() );
       CPPUNIT_ASSERT_EQUAL( samples, sampleNumber );
     }
     { // Use Case:
       // constructor with fixed size of samples, sized correctly
       // call addSamples with data that will underflow the buffer 
       // Expect:
       // addSamples to complete OK, with space left in the buffer for more samples
       unsigned samples = nSamples + 2;
       DedispersionBuffer b( samples,sampleSize );
       sampleNumber = 0;
       CPPUNIT_ASSERT_EQUAL( (unsigned)2 , b.addSamples( &data, noise, &sampleNumber ));
       CPPUNIT_ASSERT_EQUAL((unsigned int)2, b.spaceRemaining() );
       CPPUNIT_ASSERT_EQUAL( nSamples , sampleNumber );
     }
}

void DedispersionBufferTest::test_copy() {
     unsigned nSamples = 10;
     unsigned nChannels = 10;
     unsigned nPolarisations=2;
     unsigned nSubbands=2;
     unsigned sampleSize = nSubbands * nPolarisations * nChannels;
     SpectrumDataSetStokes data;
     WeightedSpectrumDataSet wdata;
     QVector<float> noise;
     _fillData( &data, 1.0 );
     SpectrumDataSetStokes data2;
     WeightedSpectrumDataSet wdata2;
     _fillData( &data2, 1000.0 );
     data.resize( nSamples, nSubbands, nPolarisations, nChannels );
     data2.resize( nSamples, nSubbands, nPolarisations, nChannels );
     SpectrumDataSetStokes dataRef(data); // data should not change
     SpectrumDataSetStokes data2Ref(data2); // data should not change
     { // Use Case:
       // copy from on identical buffer to the next, zero offset
       // single datablob, complete
       // Expect:
       // Same datablob in each buffer, same number of samples in each
       DedispersionBuffer b1( nSamples,sampleSize );
       unsigned nSamp = 0;
       b1.addSamples( &wdata, noise, &nSamp );
       CPPUNIT_ASSERT_EQUAL( 1, b1.inputDataBlobs().size() );
       DedispersionBuffer b2( nSamples,sampleSize );
       b1.copy(&b2,noise,nSamples);
       CPPUNIT_ASSERT_EQUAL( 1, b2.inputDataBlobs().size() );
       CPPUNIT_ASSERT_EQUAL( b1.numberOfSamples(), b2.numberOfSamples() );
       CPPUNIT_ASSERT_EQUAL( &data, b2.inputDataBlobs()[0] );
       // verify data has not been changed
       CPPUNIT_ASSERT( data == dataRef );
     }
     { // Use Case:
       // copy a complete single blob sampleset when mutliple blobs are in the buffer
       // Expect:
       // copy only the last sample
       DedispersionBuffer b1( 2*nSamples,sampleSize );
       unsigned nSamp = 0;
       b1.addSamples( &wdata, noise, &nSamp );
       nSamp = 0;
       b1.addSamples( &wdata2, noise, &nSamp );
       CPPUNIT_ASSERT_EQUAL( 2 * nSamples , b1.numberOfSamples() );
       DedispersionBuffer b2( 2 * nSamples,sampleSize );
       b1.copy(&b2, noise, nSamples);
       CPPUNIT_ASSERT_EQUAL( 2, b1.inputDataBlobs().size() );
       CPPUNIT_ASSERT_EQUAL( 1, b2.inputDataBlobs().size() );
       CPPUNIT_ASSERT_EQUAL( 2 * nSamples , b1.numberOfSamples() );
       CPPUNIT_ASSERT_EQUAL( &data2, b2.inputDataBlobs()[0] );
       // verify data has not been changed
       CPPUNIT_ASSERT( data == dataRef );
       CPPUNIT_ASSERT( data2 == data2Ref );
     }
     { // Use Case:
       // Copy a number of samples that do not correspond to 
       // an exact buffer.
       // Expect:
       // expect correct number of samples, samples taken from
       // the end of the first blob data.
       DedispersionBuffer b1( 2 * nSamples,sampleSize );
       unsigned nSamp = 0;
       b1.addSamples( &wdata, noise, &nSamp );
       nSamp = 0;
       b1.addSamples( &wdata2, noise, &nSamp );
       CPPUNIT_ASSERT_EQUAL( 2 * nSamples , b1.numberOfSamples() );
       DedispersionBuffer b2( 2 * nSamples ,sampleSize );
       b1.copy(&b2, noise, nSamples + 1 );
       CPPUNIT_ASSERT_EQUAL( 2, b2.inputDataBlobs().size() );
       CPPUNIT_ASSERT_EQUAL( nSamples + 1, b2.numberOfSamples() );
       CPPUNIT_ASSERT_EQUAL( &data, b2.inputDataBlobs()[0] );
       CPPUNIT_ASSERT_EQUAL( &data2, b2.inputDataBlobs()[1] );
       CPPUNIT_ASSERT( data == dataRef );
       CPPUNIT_ASSERT( data2 == data2Ref );
     }

}

void DedispersionBufferTest::_fillData( SpectrumDataSetStokes* spectrumData, float start ) {
    // each sample is filled with the sample number, with an offset of start
    unsigned samples = spectrumData->nTimeBlocks();
    unsigned nSubbands = spectrumData->nSubbands();
    unsigned nChannels= spectrumData->nChannels();
    for(unsigned t = 0; t < samples; ++t) {
        for (unsigned s = 0; s < nSubbands; ++s) {
            float* data = spectrumData->spectrumData(t, s, 0);
            for (unsigned c = 0; c < nChannels; ++c) {
                data[c]=start;
            }
        }
        ++start;
    }
}

} // namespace ampp
} // namespace pelican
