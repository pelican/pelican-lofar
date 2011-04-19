#include "FilterBankAdapterTest.h"
#include "pelican/utility/TestConfig.h"
#include "pelican/core/test/AdapterTester.h"
#include "SpectrumDataSet.h"
#include "FilterBankAdapter.h"

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( FilterBankAdapterTest );
// class FilterBankAdapterTest 
FilterBankAdapterTest::FilterBankAdapterTest()
    : CppUnit::TestFixture()
{
}

FilterBankAdapterTest::~FilterBankAdapterTest()
{
}

void FilterBankAdapterTest::setUp()
{
}

void FilterBankAdapterTest::tearDown()
{
}

void FilterBankAdapterTest::test_readFile()
{
     try {
     { // Use Case:
       // read in a file with a header
       // requesty to read a single block
       // Expect:
       // header and block to be read in and DataBlob filled
       // TestConfig config("StreamDataSet.xml", "lib");
       // create the DataBlob
       SpectrumDataSetStokes* blob = new SpectrumDataSetStokes();
       QString xml = "";
       pelican::test::AdapterTester tester("FilterBankAdapter", xml);
       tester.setDataFile(pelican::test::TestConfig::findTestFile("testData.dat","lib"));
       tester.execute(blob);

       CPPUNIT_ASSERT_EQUAL( (unsigned int)1   , blob->nPolarisations() );
       //CPPUNIT_ASSERT_EQUAL( (unsigned int)496 , blob->nChannels() );
       
       delete blob;
     }
     }
     catch( QString& e ) {
        CPPUNIT_FAIL( e.toStdString() );
     }
}

} // namespace lofar
} // namespace pelican
