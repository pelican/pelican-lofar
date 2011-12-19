#include "GPU_ManagerTest.h"
#include "GPU_Manager.h"
#include "GPU_Job.h"
#include "GPU_TestCard.h"
#include <boost/bind.hpp>


namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( GPU_ManagerTest );
/**
 *@details GPU_ManagerTest 
 */
GPU_ManagerTest::GPU_ManagerTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
GPU_ManagerTest::~GPU_ManagerTest()
{
}

void GPU_ManagerTest::setUp()
{
}

void GPU_ManagerTest::tearDown()
{
}

void GPU_ManagerTest::test_submit()
{
     // Use Case:
     // Single gpu card
     // Sumbit a single job 
     // - ensure it is sent for processsing
     // Submit a second job
     // - ensure the job is put on the queue
     // - ensure that the finished signal is emitted on the job
     //   completion and the second job is then processed
     GPU_Manager m; // a single Test card
     GPU_TestCard* card = new GPU_TestCard;
     m.addResource( card );
     CPPUNIT_ASSERT_EQUAL( 1, m.freeResources() );
     GPU_Job testJob1;
     GPU_Job testJob2;
     testJob2.addCallBack( boost::bind( &GPU_ManagerTest::callBackTest,this ) );
     _callbackCount = 0;
     m.submit(&testJob1);
     do{ sleep(1); } while( testJob1.status() == GPU_Job::Queued );
     CPPUNIT_ASSERT_EQUAL( 0, m.jobsQueued() );

     CPPUNIT_ASSERT_EQUAL( &testJob1, card->currentJob() );
     CPPUNIT_ASSERT_EQUAL( 0, m.freeResources() );
     m.submit(&testJob2);
     CPPUNIT_ASSERT_EQUAL( 1, m.jobsQueued() );

     CPPUNIT_ASSERT_EQUAL( GPU_Job::Queued, testJob2.status() );
     card->completeJob();
     do{ sleep(1); } while( testJob2.status() == GPU_Job::Queued );
     CPPUNIT_ASSERT_EQUAL( &testJob2, card->currentJob() );
     card->completeJob();
     do{ sleep(1); } while( testJob2.status() != GPU_Job::Finished );
     CPPUNIT_ASSERT_EQUAL( 1, _callbackCount );
     CPPUNIT_ASSERT_EQUAL( 1, m.freeResources() );
     CPPUNIT_ASSERT_EQUAL( 0, m.jobsQueued() );

}

void GPU_ManagerTest::callBackTest() {
     ++_callbackCount;
}

void GPU_ManagerTest::test_submitMultiCards()
{
     // Use Case:
     // Two gpu cards and two jobs
     // Sumbit a single job 
     // - ensure it is sent for processsing
     // Submit a second job
     // - ensure the job is put on the queue
     // - ensure that the finished signal is emitted on the job
     //   completion and the second job is then processed
     GPU_Manager m; // a single Test card
     GPU_TestCard* card1 = new GPU_TestCard;
     GPU_TestCard* card2 = new GPU_TestCard;
     m.addResource( card1 );
     m.addResource( card2 );
     CPPUNIT_ASSERT_EQUAL( 2, m.freeResources() );

     GPU_Job testJob1;
     GPU_Job testJob2;
     m.submit(&testJob1);
     m.submit(&testJob2);

     do{ sleep(1); } while( testJob1.status() == GPU_Job::Queued );
     do{ sleep(1); } while( testJob2.status() == GPU_Job::Queued );
     CPPUNIT_ASSERT_EQUAL( &testJob1, card1->currentJob() );
     CPPUNIT_ASSERT_EQUAL( &testJob2, card2->currentJob() );
     CPPUNIT_ASSERT_EQUAL( 0, m.freeResources() );
     card2->completeJob();
     do{ sleep(1); } while( testJob2.status() != GPU_Job::Finished );
     CPPUNIT_ASSERT_EQUAL( 1, m.freeResources() );
     card1->completeJob();
     do{ sleep(1); } while( testJob1.status() != GPU_Job::Finished );
     CPPUNIT_ASSERT_EQUAL( 2, m.freeResources() );
     CPPUNIT_ASSERT_EQUAL( 0, m.jobsQueued() );

}
} // namespace lofar
} // namespace pelican
