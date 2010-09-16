#include "RFI_ClipperTest.h"
#include "RFI_Clipper.h"
#include "SpectrumDataSet.h"
#include <iostream>


namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( RFI_ClipperTest );
/**
 *@details RFI_ClipperTest 
 */
RFI_ClipperTest::RFI_ClipperTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
RFI_ClipperTest::~RFI_ClipperTest()
{
}

void RFI_ClipperTest::setUp()
{
}

void RFI_ClipperTest::tearDown()
{
}


void RFI_ClipperTest::test_goodData()
{
    try {
    // Use Case:
    // Data has no significant pikes
    // Expect:
    // Pass through unchanged
    ConfigNode config;
    RFI_Clipper rfi(config);

    SpectrumDataSetStokes data;
    SpectrumDataSetStokes expect;
    _initSubbandData(data, expect, 4);
    rfi.run(&data);
    CPPUNIT_ASSERT_EQUAL( 0,  _diff(data, expect).size() );
    }
    catch( QString s )
    {
        CPPUNIT_FAIL(s.toStdString());
    }
}

void RFI_ClipperTest::test_badSubband()
{
    try {
    // Use Case:
    // Data has a single subband which is bad
    // Expect:
    // Bad Subband is reduced to 0, all other values pass OK
    ConfigNode config;
    RFI_Clipper rfi(config);

    SpectrumDataSetStokes data;
    SpectrumDataSetStokes expect;
    int nChannels = 16;
    _initSubbandData( data, expect, 31, nChannels );
    int badBlock = 0;
    int badSubband = 0;
    int badPol = 0;

    float* d = data.spectrumData(badBlock,badSubband,badPol); 
    float* d2 = expect.spectrumData(badBlock,badSubband,badPol); 
    for(int channel=0; channel < nChannels; ++channel)
    {
        d[channel] *= 3;
        d2[channel] = 0;
    }

    rfi.run(&data);
    QList<RFI_ClipperTest::StokesIndex> different = _diff(expect,data);
    if( different.size() > 0  )
    {
        foreach( const StokesIndex& i, different )
        {
            std::cout << "badSubband: block=" << i.block << " subband=" << i.subband << " pol=" << i.polarisation << "channel=" << i.channel <<std::endl;
        }
        CPPUNIT_ASSERT_EQUAL( 0 ,  different.size() );
    }
    //CPPUNIT_ASSERT_EQUAL( 0 ,  _diff(expect ,data).size() );
    }
    catch( QString s )
    {
        CPPUNIT_FAIL(s.toStdString());
    }
}

void RFI_ClipperTest::test_badChannel()
{
    try {
    // Use Case:
    // Data has a single channel in a single subband which is bad
    // Expect:
    // Bad channel is reduced to 0, all other values are OK
    ConfigNode config;
    RFI_Clipper rfi(config);

    SpectrumDataSetStokes data;
    SpectrumDataSetStokes expect;
    _initSubbandData(data, expect, 31, 16);

    // put in a bad channel
    int badBlock = 0;
    int badSubband = 1;
    int badChannel = 2;
    int badPol = 0;
    float* d = data.spectrumData(badBlock,badSubband,badPol); 
    float* e = expect.spectrumData(badBlock,badSubband,badPol); 
    d[badChannel] *= 3; // should not catch below this
    e[badChannel] = 0;

    rfi.run(&data);
    QList<RFI_ClipperTest::StokesIndex> different = _diff(expect,data);
    if( different.size() > 0  )
    {
        foreach( const StokesIndex& i, different )
        {
            std::cout << "badChannel: block=" << i.block << " subband=" << i.subband << " pol=" << i.polarisation << "channel=" << i.channel <<std::endl;
        }
        CPPUNIT_ASSERT_EQUAL( 0 ,  different.size() );
    }
    }
    catch( QString s )
    {
        CPPUNIT_FAIL(s.toStdString());
    }
}

void  RFI_ClipperTest::_initSubbandData( SpectrumDataSetStokes& primary, SpectrumDataSetStokes& shifted, int numberOfSubbands, int numberOfChannels )
{
    int numberOfBlocks = 10;
    int numberOfPolarisations = 1;
    float value = (numberOfChannels/2.0);
    int maxRange = value - 1;

    //float av = 0;
    primary.resize( numberOfBlocks, numberOfSubbands, numberOfPolarisations, numberOfChannels );
    for( int block=0; block < numberOfBlocks; ++block ) {
        for( int subband=0; subband < numberOfSubbands; ++subband ) {
            for( int polarisation=0; polarisation < numberOfPolarisations; ++polarisation ) {
                float* ptr = primary.spectrumData( block, subband, polarisation );
                for( int i=0; i < numberOfChannels; ++i ) {
                    ptr[i] = value + rand()%maxRange;
                    //av += ptr[i];
                }
            }
        }
    }
    shifted = primary; // no longer need to shift
    // generate the shifted dataseto
    /*
    for( int block=0; block < numberOfBlocks; ++block ) {
        for( int subband=0; subband < numberOfSubbands; ++subband ) {
            for( int polarisation=0; polarisation < numberOfPolarisations; ++polarisation ) {
                float* ptr = primary.spectrumData( block, subband, polarisation );
                float* sptr = shifted.spectrumData( block, subband, polarisation );
                for( int i=0; i < numberOfChannels; ++i ) {
                    sptr[i] = ptr[0];
                    av += ptr[i];
                }
            }
        }
    }
    */
}

// N.B. assumes they are the same dimension
// returns a list of all the indices that differ
QList<RFI_ClipperTest::StokesIndex> RFI_ClipperTest::_diff(const SpectrumDataSetStokes& a, 
                           const SpectrumDataSetStokes& b)
{
    QList<StokesIndex> diff;
    if( a.size() != b.size() )
        throw(QString(
               "RFI_ClipperTest::_diff(): called with different size objects")
             );
    int numberOfBlocks = a.nTimeBlocks();
    int numberOfPolarisations = a.nPolarisations();
    int numberOfSubbands = a.nSubbands();
    int numberOfChannels = a.nChannels();
    for( int block=0; block < numberOfBlocks; ++block ) {
        for( int subband=0; subband < numberOfSubbands; ++subband ) {
            for( int polarisation=0; polarisation < numberOfPolarisations; 
                    ++polarisation ) {
                const float* ptra = a.spectrumData( block, subband, polarisation );
                const float* ptrb = b.spectrumData( block, subband, polarisation );
                for( int i=0; i < numberOfChannels; ++i ) {
                    if( ptra[i] != ptrb[i] )
                    {
                        std::cout << "subband=" << subband << "channel=" << i << " a=" << ptra[i] << " b=" <<  ptrb[i] << std::endl;
                        StokesIndex index(block,subband,polarisation,i);
                        diff.append( index );
                    }
                }
            }
        }
    }
    return diff;
}

void RFI_ClipperTest::dump(const SpectrumDataSetStokes a)
{
    std::cout << "-----------------------------------------" << std::endl;
    int numberOfBlocks = a.nTimeBlocks();
    int numberOfPolarisations = a.nPolarisations();
    int numberOfSubbands = a.nSubbands();
    int numberOfChannels = a.nChannels();
    for( int block=0; block < numberOfBlocks; ++block ) {
        for( int subband=0; subband < numberOfSubbands; ++subband ) {
            for( int polarisation=0; polarisation < numberOfPolarisations; 
                    ++polarisation ) {
                const float* ptra = a.spectrumData( block, subband, polarisation );
                for( int i=0; i < numberOfChannels; ++i ) {
                    std::cout << " dump(): block=" << block << " subband=" << subband << "channel=" << i << " a=" << ptra[i] << std::endl;
                }
            }
        }
    }
    std::cout << "-----------------------------------------" << std::endl;
}

} // namespace lofar
} // namespace pelican
