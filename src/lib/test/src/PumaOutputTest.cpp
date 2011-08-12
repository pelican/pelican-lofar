#include "PumaOutputTest.h"
#include <QtCore/QFile>
#include <QtCore/QDir>
#include <QtCore/QCoreApplication>
#include "PumaOutput.h"
#include "pelican/utility/ClientTestServer.h"
#include "pelican/utility/ConfigNode.h"


namespace pelican {

namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( PumaOutputTest );
/**
 *@details PumaOutputTest
 */
PumaOutputTest::PumaOutputTest()
    : CppUnit::TestFixture()
{
}

/**
 *@details
 */
PumaOutputTest::~PumaOutputTest()
{
}

void PumaOutputTest::setUp()
{
    _filename = QDir::tempPath() + "/_PumaOutputTest_";
#ifdef QT_VERSION >= 0x040400
       _filename += QString().setNum( QCoreApplication::applicationPid() );
#endif
    int argc = 1;
    char *argv[] = {(char*)"lofartest"};
    _app = new QCoreApplication(argc,argv);
}

void PumaOutputTest::tearDown()
{
    QFile f(_filename);
    if( f.exists() )
        f.remove();
    delete _app;
}

void PumaOutputTest::test_configuration()
{
    try {
    { // Use Case:
      // No Configuration given
      // Expect:
      // Be able to construct the object
      QString xml;
      ConfigNode c;
      PumaOutput out( c );
    }
    { // Use Case:
      // Configuration with a file
      // Expect:
      // File to be generated
      QString xml = "<PumaOutput>\n"
                    "<file name=\"" + _filename + "\" />\n"
                    "</PumaOutput>";
      ConfigNode c;
      c.setFromString(xml);
      {
          PumaOutput out( c );
          out.send("data", &_dummyData );
      }
      QFile f(_filename);
      CPPUNIT_ASSERT( f.exists() );
      CPPUNIT_ASSERT_EQUAL( _dummyData.size(), (int)f.size() );
    }
    { // Use Case:
      // Configuration with a host
      // Expect:
      // Attempt to connect to host
      ClientTestServer testHost;
      QString xml = "<PumaOutput>\n"
                    "<conncetion host=\"" + testHost.hostname() + "\" port=\""
                    + testHost.port() + "\" />\n"
                    "</PumaOutput>";
      ConfigNode c;
      c.setFromString(xml);
      PumaOutput out( c );
      out.send("data", &_dummyData );
      CPPUNIT_ASSERT_EQUAL( _dummyData.size(), (int)testHost.dataReceived().size() );
    }
    }
    catch( QString& s )
    {
        CPPUNIT_FAIL(s.toStdString());
    }
}

} // namespace lofar
} // namespace pelican
