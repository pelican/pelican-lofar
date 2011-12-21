#include "GPU_NVidiaTest.h"
#include "GPU_NVidia.h"
#include "GPU_Manager.h"
#include "GPU_Job.h"
#include "GPU_MemoryMap.h"
#include "TestCudaVectorAdd.h"
#include "GPU_NVidiaConfiguration.h"
#include <string>


namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( GPU_NVidiaTest );
/**
 *@details GPU_NVidiaTest 
 */
GPU_NVidiaTest::GPU_NVidiaTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
GPU_NVidiaTest::~GPU_NVidiaTest()
{
}

void GPU_NVidiaTest::setUp()
{
}

void GPU_NVidiaTest::tearDown()
{
}

void GPU_NVidiaTest::test_managedCard()
{
    GPU_Manager m;
    GPU_NVidia::initialiseResources( &m );
    if( m.freeResources() == 0 ) {
       std::string msg="unable to run test - no NVidia cards available";
       CPPUNIT_FAIL( msg );
    }
    // set up a GPU vector addition job
    int size = 2 ;
    std::vector<float> vec1(size);
    vec1[0]=1.0;
    vec1[1]=2.0;
    std::vector<float> vec2(size);
    vec2[0]=100.0;
    vec2[1]=102.0;
    std::vector<float> result(size);
    CPPUNIT_ASSERT( vec1[0] + vec2[0] != result[0] );
    CPPUNIT_ASSERT( vec1[1] + vec2[1] != result[1] );
    GPU_Job job;
    GPU_MemoryMap vec1map( vec1 );
    CPPUNIT_ASSERT( vec1[0] == 1.0  );
    GPU_MemoryMap vec2map( vec2 );
    GPU_MemoryMap resultmap( result );
    TestCudaVectorAdd testKernel;
    GPU_NVidiaConfiguration config;
    config.addInputMap( vec1map );
    config.addInputMap( vec2map );
    config.addOutputMap( resultmap );
    testKernel.setConfiguration( config );
    job.addKernel( &testKernel );
    m.submit(&job);
    job.wait();
    CPPUNIT_ASSERT_EQUAL( vec1[0] + vec2[0] , result[0] );
    CPPUNIT_ASSERT_EQUAL( vec1[1] + vec2[1] , result[1] );
    CPPUNIT_ASSERT_EQUAL( GPU_Job::Finished, job.status() );
}

} // namespace lofar
} // namespace pelican
