#ifndef FILTERBANKADAPTERTEST_H
#define FILTERBANKADAPTERTEST_H

#include <cppunit/extensions/HelperMacros.h>

/* Class      : FilterBankAdapterTest
 * Revision   : $Rev$ 
 * Description:
 *     Unit test for the FilterBankAdpater
 */

namespace pelican {
namespace lofar {

class FilterBankAdapterTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( FilterBankAdapterTest );
        CPPUNIT_TEST( test_readFile );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_readFile();

    public:
        FilterBankAdapterTest(  );
        ~FilterBankAdapterTest();

    private:
};

} // namespace lofar
} // namespace pelican
#endif // FILTERBANKADAPTERTEST_H 
