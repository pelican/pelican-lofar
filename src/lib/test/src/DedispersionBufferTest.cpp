#include "DedispersionBufferTest.h"
#include "DedispersionBuffer.h"
#include "WeightedSpectrumDataSet.h"
#include "SpectrumDataSet.h"


namespace pelican {

namespace lofar {

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
     SpectrumDataSet<float> spectrumData;
     spectrumData.resize( nSamples, nSubbands, nPolarisations, nChannels );
     WeightedSpectrumDataSet data( &spectrumData );
     { // Use Case:
       // Default constructor
       // Expect:
       // Zero size
       // addSamples to do nothing

       DedispersionBuffer b;
       CPPUNIT_ASSERT_EQUAL(0, (int)b.size() );
       CPPUNIT_ASSERT_EQUAL((unsigned int)0, b.numSamples() );
       CPPUNIT_ASSERT_EQUAL((unsigned int)0, b.spaceRemaining() );
       CPPUNIT_ASSERT_EQUAL( b.maxSamples() , b.addSamples( &data, &sampleNumber ));
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
       CPPUNIT_ASSERT_EQUAL( (unsigned)0 , b.addSamples( &data, &sampleNumber ));
       CPPUNIT_ASSERT_EQUAL( nSamples, b.numSamples() );
       CPPUNIT_ASSERT_EQUAL((unsigned int)0, b.spaceRemaining() );
       CPPUNIT_ASSERT_EQUAL( nSamples, sampleNumber );
       // we should not be able to add any more data as the buffer is full
       sampleNumber = 0;
       CPPUNIT_ASSERT_EQUAL( (unsigned)0 , b.addSamples( &data, &sampleNumber ));
       CPPUNIT_ASSERT_EQUAL( (unsigned)0, sampleNumber );
       // clear it and we should be able to use it again
       b.clear();
       CPPUNIT_ASSERT_EQUAL((unsigned int)0, b.numSamples() );
       CPPUNIT_ASSERT_EQUAL((unsigned int)10, b.spaceRemaining() );
       CPPUNIT_ASSERT_EQUAL( (unsigned)0 , b.addSamples( &data, &sampleNumber ));
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
       CPPUNIT_ASSERT_EQUAL( (unsigned)0 , b.addSamples( &data, &sampleNumber ));
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
       CPPUNIT_ASSERT_EQUAL( (unsigned)2 , b.addSamples( &data, &sampleNumber ));
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
     SpectrumDataSet<float> spectrumData;
     spectrumData.resize( nSamples, nSubbands, nPolarisations, nChannels );
     WeightedSpectrumDataSet data( &spectrumData );
     { // Use Case:
       // copy from on identical buffer to the next, zero offset
       // Expect:
       // identical data throughout
       DedispersionBuffer b1( nSamples,sampleSize );
       DedispersionBuffer b2( nSamples,sampleSize );
       b2.copy(&b1,0);
       
       //TODO verify data pattern
     }
     { // Use Case:
       // copy from on identical buffer to the next, last byte
       // Expect:
       // copy only the last sample
       DedispersionBuffer b1( nSamples,sampleSize );
       DedispersionBuffer b2( nSamples,sampleSize );
       b2.copy(&b1,sampleSize * nSamples- 1);
       //TODO verify data pattern
     }
     { // Use Case:
       // copy from on identical buffer to the next, end of data offset
       // Expect:
       // copy nothing, without crashing
       DedispersionBuffer b1( nSamples,sampleSize );
       DedispersionBuffer b2( nSamples,sampleSize );
       b2.copy(&b1, sampleSize*nSamples);
       //TODO verify data pattern
     }
}

} // namespace lofar
} // namespace pelican
