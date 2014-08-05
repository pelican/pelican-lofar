#include "GPU_MemoryMapTest.h"
#include "GPU_MemoryMap.h"


namespace pelican {

namespace ampp {

CPPUNIT_TEST_SUITE_REGISTRATION( GPU_MemoryMapTest );
/**
 *@details GPU_MemoryMapTest 
 */
GPU_MemoryMapTest::GPU_MemoryMapTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
GPU_MemoryMapTest::~GPU_MemoryMapTest()
{
}

void GPU_MemoryMapTest::setUp()
{
}

void GPU_MemoryMapTest::tearDown()
{
}

void GPU_MemoryMapTest::test_method()
{
     int testVar = 100;
     GPU_MemoryMap m( &testVar, sizeof(testVar) );
     CPPUNIT_ASSERT_EQUAL( testVar, m.value<int>() );
}

} // namespace ampp
} // namespace pelican
