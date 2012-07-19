#ifndef H5_LOFARBFDATAWRITERTEST_H
#define H5_LOFARBFDATAWRITERTEST_H

#include <cppunit/extensions/HelperMacros.h>
#include <QDir>

/**
 * @file H5_LofarBFDataWriterTest.h
 */

namespace pelican {

namespace lofar {

/**
 * @class H5_LofarBFDataWriterTest
 *  
 * @brief
 * 
 * @details
 * 
 */

class H5_LofarBFDataWriterTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( H5_LofarBFDataWriterTest );
        CPPUNIT_TEST( test_method );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_method();

    public:
        H5_LofarBFDataWriterTest(  );
        ~H5_LofarBFDataWriterTest();

    private:
        QString _fileDir;
        QString _fullFileDir;
};

} // namespace lofar
} // namespace pelican
#endif // H5_LOFARBFDATAWRITERTEST_H 
