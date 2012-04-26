#include "LockingContainerTest.h"
#include "LockingContainer.hpp"


namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( LockingContainerTest );
/**
 *@details LockingContainerTest 
 */
LockingContainerTest::LockingContainerTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
LockingContainerTest::~LockingContainerTest()
{
}

void LockingContainerTest::setUp()
{
}

void LockingContainerTest::tearDown()
{
}

void LockingContainerTest::test_method()
{
    QList<int> buf; buf << 1 << 2 << 3 << 4;
    LockingContainer<int> buffer(&buf);
    int* b1 = buffer.next();
    CPPUNIT_ASSERT_EQUAL(1, *b1 );
    int* b2 = buffer.next();
    CPPUNIT_ASSERT_EQUAL(2, *b2 );
    int* b3 = buffer.next();
    CPPUNIT_ASSERT_EQUAL(3, *b3 );
    int* b4 = buffer.next();
    CPPUNIT_ASSERT_EQUAL(4, *b4 );
    buffer.unlock( b1 );
    b1 = 0;
    b1 = buffer.next();
    CPPUNIT_ASSERT_EQUAL(1, *b1 );
    buffer.unlock( b3 );
    b3 = 0;
    b3 = buffer.next();
    CPPUNIT_ASSERT_EQUAL(3, *b3 );
}

} // namespace lofar
} // namespace pelican
