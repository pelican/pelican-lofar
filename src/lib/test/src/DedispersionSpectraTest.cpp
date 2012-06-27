#include "DedispersionSpectraTest.h"
#include "DedispersionSpectra.h"


namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( DedispersionSpectraTest );
/**
 *@details DedispersionSpectraTest 
 */
DedispersionSpectraTest::DedispersionSpectraTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
DedispersionSpectraTest::~DedispersionSpectraTest()
{
}

void DedispersionSpectraTest::setUp()
{
}

void DedispersionSpectraTest::tearDown()
{
}

void DedispersionSpectraTest::test_dmIndex()
{
    unsigned timebins=100;
    unsigned dedispersionBins = 100;
    float dedispersionBinStart = 0.1;
    float dedispersionBinWidth = 0.2;
    float maxDmAmplitude = 1001.1;
    int dataSize = timebins * dedispersionBins;
    DedispersionSpectra spectra;
    spectra.resize( timebins, dedispersionBins, dedispersionBinStart,
                    dedispersionBinWidth );
    CPPUNIT_ASSERT_EQUAL( dataSize, spectra.data().size() );
    spectra.data()[ dataSize - 1 ] = maxDmAmplitude;
    float max = dedispersionBinStart + dedispersionBinWidth * (dedispersionBins -1 );
    CPPUNIT_ASSERT_DOUBLES_EQUAL( max, spectra.dmMax(), 0.0001 );
    CPPUNIT_ASSERT_EQUAL( (int)dedispersionBins, spectra.dmBins() );
    //CPPUNIT_ASSERT_EQUAL( maxDmAmplitude , spectra.dm( max ) );
    CPPUNIT_ASSERT_EQUAL( dedispersionBins - 1, spectra.dmIndex( max ) );
    CPPUNIT_ASSERT_EQUAL( (int)timebins, spectra.timeSamples() );
    // try accessing the max amplitude
    // expect not to fail
    CPPUNIT_ASSERT_EQUAL( maxDmAmplitude, spectra.dmAmplitude( timebins, max ) );
}

} // namespace lofar
} // namespace pelican
