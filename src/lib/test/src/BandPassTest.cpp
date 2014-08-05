#include "BandPassTest.h"
#include "BinMap.h"
#include "BandPass.h"
#include <QtCore/QVector>


namespace pelican {

namespace ampp {

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
     float start=137.304688;
     float width=-0.0007628;
     map.setStart(start);
     map.setBinWidth(width);
     QVector<float> params;
     params << 4460.84130843 << -24.8135957376;
     bp.setData(map, params);
     float rms = 44.3413175435;
     bp.setRMS(rms);
     float median=1128.9281113;
     bp.setMedian(median);

     // add a kill zone
     float killStart= start + 3*width;
     float killEnd= start + 15*width;
     bp.killBand( killStart, killEnd );

     float a = bp.intensityOfBin(0);

     {  // Use Case:
        // rebin to twice as many bins over the same range
        BinMap map(7936*2);
        map.setStart(start);
        map.setBinWidth(width/2.0);
        bp.reBin(map);
        CPPUNIT_ASSERT( bp.intensity( killStart) < 0.00001 );
        CPPUNIT_ASSERT( bp.intensity( killEnd) < 0.00001 );
        CPPUNIT_ASSERT( bp.intensityOfBin( 15*2 ) < 0.00001 );
        CPPUNIT_ASSERT_DOUBLES_EQUAL( median/2.0 , bp.median() , 0.000001 );
        CPPUNIT_ASSERT_DOUBLES_EQUAL( rms * std::sqrt(2.0) , bp.rms() , 0.000001 );
        CPPUNIT_ASSERT_DOUBLES_EQUAL( a/2.0 , bp.intensityOfBin(0) , 0.001 );
     }
     {  // Use Case:
        // rebin to twice as many bins over the same range
     }

}

void BandPassTest::test_setMedian()
{
     // setup the reference bandpass
     BandPass bp;
     BinMap map(7936);
     float start=137.304688;
     float width=-0.0007628;
     map.setStart(start);
     map.setBinWidth(width);
     QVector<float> params;
     params << 4460.84130843 << -24.8135957376;
     bp.setData(map, params);
     float rms = 44.3413175435;
     bp.setRMS(rms);
     float median=bp.median();

     {  // Use Case:
        // set medium on the primary map
        // expect values to be shifted
        float halfMedian = median/2.0;
        bp.setMedian(halfMedian);
        CPPUNIT_ASSERT_DOUBLES_EQUAL( halfMedian , bp.median() , 0.000001 );

        // test params have been updated correctly
        BandPass bp2; // use to calculate a median
        bp2.setData(map,bp.params());
        CPPUNIT_ASSERT_DOUBLES_EQUAL( bp2.median() , bp.median() , 0.001 );
     }
}

} // namespace ampp
} // namespace pelican
