#include "BinMapTest.h"
#include "BinMap.h"


namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( BinMapTest );
/**
 *@details BinMapTest 
 */
BinMapTest::BinMapTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
BinMapTest::~BinMapTest()
{
}

void BinMapTest::setUp()
{
}

void BinMapTest::tearDown()
{
}

void BinMapTest::test_bin()
{
     BinMap map(32*256);
     double start = 142.779541;
     double width = 0.006104;
     map.setStart(start);
     map.setBinWidth(width);
     CPPUNIT_ASSERT_DOUBLES_EQUAL( start, map.binAssignmentNumber(0), 0.00001 );
     CPPUNIT_ASSERT_DOUBLES_EQUAL( start - width/2.0 , map.binStart(0), 0.00001 );
     CPPUNIT_ASSERT_DOUBLES_EQUAL( start + width/2.0 , map.binEnd(0), 0.00001 );
     CPPUNIT_ASSERT_DOUBLES_EQUAL( start + width , map.binAssignmentNumber(1), 0.00001 );
     CPPUNIT_ASSERT_DOUBLES_EQUAL( start + 2*width , map.binAssignmentNumber(2), 0.00001 );
     CPPUNIT_ASSERT_EQUAL( 0 , map.binIndex(start) );
     CPPUNIT_ASSERT_EQUAL( 1 , map.binIndex(start + width) );
}

} // namespace lofar
} // namespace pelican
