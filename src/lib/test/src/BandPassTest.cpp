#include "BandPassTest.h"
#include "BinMap.h"
#include "BandPass.h"
#include <QVector>


namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( BandPassTest );
/**
 *@details BandPassTest 
 */
BandPassTest::BandPassTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
BandPassTest::~BandPassTest()
{
}

void BandPassTest::setUp()
{
}

void BandPassTest::tearDown()
{
}

void BandPassTest::test_reBin()
{
     // setup the reference bandpass
     BandPass bp;
     BinMap map(7936);
     map.setStart(137.304688);
     map.setBinWidth(-0.0007628);
     QVector<float> params;
     params << 4460.84130843 << -24.8135957376;
     bp.setData(map, params);
     float rms = 44.3413175435;
     bp.setRMS(rms);
     float median=1128.9281113;
     bp.setMedian(median);

     float a = bp.intensityOfBin(0);
     
     {  // Use Case:
        // rebin to twice as many bins over the same range
        BinMap map(7936*2);
        map.setStart(137.304688);
        map.setBinWidth(-0.0007628/2.0);
        bp.reBin(map);
        CPPUNIT_ASSERT_DOUBLES_EQUAL( median/2.0 , bp.median() , 0.000001 );
        CPPUNIT_ASSERT_DOUBLES_EQUAL( rms * std::sqrt(2.0) , bp.rms() , 0.000001 );
        CPPUNIT_ASSERT_DOUBLES_EQUAL( a/2.0 , bp.intensityOfBin(0) , 0.001 );
     }
     {  // Use Case:
        // rebin to twice as many bins over the same range
     }

}

} // namespace lofar
} // namespace pelican
