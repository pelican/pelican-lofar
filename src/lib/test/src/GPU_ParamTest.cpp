#include "GPU_ParamTest.h"
#include "GPU_Param.h"
#include <QVector>


namespace pelican {

namespace ampp {

CPPUNIT_TEST_SUITE_REGISTRATION( GPU_ParamTest );
/**
 *@details GPU_ParamTest 
 */
GPU_ParamTest::GPU_ParamTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
GPU_ParamTest::~GPU_ParamTest()
{
}

void GPU_ParamTest::setUp()
{
}

void GPU_ParamTest::tearDown()
{
}

void GPU_ParamTest::test_memoryLeak()
{
     QVector<float> data(4);
     GPU_MemoryMap map(data);
     GPU_Param param( map );
}

} // namespace ampp
} // namespace pelican
