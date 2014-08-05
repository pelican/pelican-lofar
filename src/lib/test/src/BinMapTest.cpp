#include "BinMapTest.h"
#include "BinMap.h"


namespace pelican {

namespace ampp {

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

void BinMapTest::test_hash()
{
     // Use Case:
     // two identical maps should hash to the same value
     BinMap map1(32*256);
     BinMap map2(32*256);
     unsigned int id1=map1.hash();
     unsigned int id2=map2.hash();
     CPPUNIT_ASSERT_EQUAL(id1,id2);
     map2.setStart(10030.012);
     CPPUNIT_ASSERT(id1 != map2.hash());
     map1.setStart(10030.012);
     CPPUNIT_ASSERT_EQUAL(map1.hash(), map2.hash());
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

} // namespace ampp
} // namespace pelican
