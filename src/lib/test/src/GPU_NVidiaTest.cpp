#include "GPU_NVidiaTest.h"
#include "GPU_NVidia.h"
#include "GPU_Manager.h"
#include "GPU_Job.h"
#include "GPU_MemoryMap.h"
#include "TestCudaVectorAdd.h"
#include <QVector>
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
    {
        // Use case: using std::vector
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
        testKernel.addInputMap( vec1map );
        testKernel.addConstant( vec2map ); // add as a constant
        testKernel.addOutputMap( resultmap );
        job.addKernel( &testKernel );
        m.submit(&job);
        job.wait();
        CPPUNIT_ASSERT_EQUAL( vec1[0] + vec2[0] , result[0] );
        CPPUNIT_ASSERT_EQUAL( vec1[1] + vec2[1] , result[1] );
        CPPUNIT_ASSERT_EQUAL( GPU_Job::Finished, job.status() );
    }
    {
        // Use case: using QVector
        QVector<float> vec1(size);
        vec1[0]=1.0;
        vec1[1]=2.0;
        QVector<float> vec2(size);
        vec2[0]=100.0;
        vec2[1]=102.0;
        QVector<float> result(size);
        GPU_Job job;
        GPU_MemoryMap vec1map( vec1 );
        CPPUNIT_ASSERT( vec1[0] == 1.0  );
        GPU_MemoryMap vec2map( vec2 );
        GPU_MemoryMap resultmap( result );
        TestCudaVectorAdd testKernel;
        testKernel.addInputMap( vec1map );
        testKernel.addConstant( vec2map ); // add as a constant
        testKernel.addOutputMap( resultmap );
        job.addKernel( &testKernel );
        m.submit(&job);
        job.wait();
        CPPUNIT_ASSERT_EQUAL( vec1[0] + vec2[0] , result[0] );
        CPPUNIT_ASSERT_EQUAL( vec1[1] + vec2[1] , result[1] );
        CPPUNIT_ASSERT_EQUAL( GPU_Job::Finished, job.status() );
    }
}

void GPU_NVidiaTest::test_multipleJobs() 
{
    // Use case: run multiple jobs using the same configuration
    // Expect: speed improvements due to no need to allocate GPU memory
    GPU_Manager m;
    GPU_NVidia::initialiseResources( &m );
    if( m.freeResources() == 0 ) {
       std::string msg="unable to run test - no NVidia cards available";
       CPPUNIT_FAIL( msg );
    }
    // common data between jobs
    int size = 2 ;
    std::vector<float> vec1(size);
    vec1[0]=1.0;
    vec1[1]=2.0;
    std::vector<float> vec2(size);
    vec2[0]=100.0;
    vec2[1]=102.0;
    GPU_MemoryMap vec1map( vec1 );
    GPU_MemoryMap vec2map( vec2 );

    std::vector<float> result1(size);
    std::vector<float> result2(size);
    result1[0] = 10.0;
    result2[0] = 20.0;
    GPU_MemoryMap resultmap1( result1 );
    GPU_MemoryMap resultmap2( result2 );

    TestCudaVectorAdd testKernel;
    testKernel.addConstant( vec2map ); // add as a constant
    testKernel.addInputMap( vec1map );
    testKernel.addOutputMap( resultmap1 );

    GPU_Job job1;
    GPU_Job job2;
    job1.addKernel( &testKernel );
    job2.addKernel( &testKernel );

    CPPUNIT_ASSERT( result1[0] != result2[0] );
    m.submit(&job1);
    job1.wait();
    testKernel.addInputMap( vec1map );
    testKernel.addOutputMap( resultmap2 );

    m.submit(&job2);
    job2.wait();
    CPPUNIT_ASSERT_EQUAL( GPU_Job::Finished, job1.status() );
    CPPUNIT_ASSERT_EQUAL( GPU_Job::Finished, job2.status() );
    CPPUNIT_ASSERT_EQUAL( result1[0], result2[0] );
    CPPUNIT_ASSERT_EQUAL( result1[1], result2[1] );

}

} // namespace lofar
} // namespace pelican
