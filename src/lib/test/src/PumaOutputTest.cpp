#include "PumaOutputTest.h"
#include <QFile>
#include <QDir>
#include <QCoreApplication>
#include "PumaOutput.h"
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
    _filename = QDir::tempPath() + "/_PumaOutputTest_" 
              + QString().setNum( QCoreApplication::applicationPid() );
}

void PumaOutputTest::tearDown()
{
    QFile f(_filename);
    if( f.exists() )
        f.remove();
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
/*
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
    */
    }
    }
    catch( QString& s )
    {
        CPPUNIT_FAIL(s.toStdString());
    }
}

} // namespace lofar
} // namespace pelican
