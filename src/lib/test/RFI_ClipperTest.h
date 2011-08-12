#ifndef RFI_CLIPPERTEST_H
#define RFI_CLIPPERTEST_H

#include <cppunit/extensions/HelperMacros.h>
#include <QtCore/QList>
#include <QtCore/QString>

/**
 * @file RFI_ClipperTest.h
 */

namespace pelican {
    class ConfigNode;

namespace lofar {
    class SpectrumDataSetStokes;
    class BandPass;

/**
 * @class RFI_ClipperTest
 *
 * @brief
 *   unit test for the RFI Clipper
 * @details
 *
 */

class RFI_ClipperTest : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE( RFI_ClipperTest );
        CPPUNIT_TEST( test_goodData );
        CPPUNIT_TEST( test_badChannel );
        CPPUNIT_TEST( test_badSubband );
        CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown();

        // Test Methods
        void test_goodData();
        void test_badSubband();
        void test_badChannel();

    public:
        RFI_ClipperTest(  );
        ~RFI_ClipperTest();

    private:

        class StokesIndex {
            public:
            int block;
            int subband;
            int polarisation;
            int channel;

            public:
                StokesIndex(int b,int s,int p,int c)
                    : block(b),subband(s),polarisation(p),channel(c) {};
                ~StokesIndex() {};
        };

    private:
        void dump(const SpectrumDataSetStokes a);
        void  _initSubbandData( SpectrumDataSetStokes& s, SpectrumDataSetStokes& shifted, const BandPass& bandpass, int numberOfSubbands, int numberOfChannels = 16 );
        QList<StokesIndex> _diff(const SpectrumDataSetStokes& a, const SpectrumDataSetStokes& b );
        ConfigNode testConfig(const QString& = "t191_BAND.bp");
};

} // namespace lofar
} // namespace pelicanND.bp
#endif // RFI_CLIPPERTEST_H
