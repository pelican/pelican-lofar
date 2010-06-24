#ifndef CHANNELISERPOLYPHASETEST_H
#define CHANNELISERPOLYPHASETEST_H

#include <cppunit/extensions/HelperMacros.h>
#include "pelican/utility/ConfigNode.h"

/**
 * @file ChanneliserPolyphaseTest.h
 */

namespace pelican {
namespace lofar {

/**
 * @class ChanneliserPolyphaseTest
 *
 * @brief
 *
 * @details
 * Performs unit tests on the polyphase channeliser module using the
 * CppUnit framework.
 */

class ChanneliserPolyphaseTest : public CppUnit::TestFixture
{
    public:
        ChanneliserPolyphaseTest() : CppUnit::TestFixture() {};
        ~ChanneliserPolyphaseTest() {}

    public:
        void setUp() {}
        void tearDown() {}

    public:
        /// Register test methods.
        CPPUNIT_TEST_SUITE(ChanneliserPolyphaseTest);
        CPPUNIT_TEST(test_configuration);
        CPPUNIT_TEST(test_threadAssign);
        CPPUNIT_TEST(test_updateBuffer);
        CPPUNIT_TEST(test_filter);
        CPPUNIT_TEST(test_fft);
        CPPUNIT_TEST(test_run);
        CPPUNIT_TEST(test_loadCoeffs);
//        CPPUNIT_TEST(test_makeSpectrum);
//        CPPUNIT_TEST(test_channelProfile);
        CPPUNIT_TEST_SUITE_END();

    public:
        /// Test module configuration.
        void test_configuration();

        void test_threadAssign();

        /// Test updating the delay buffer.
        void test_updateBuffer();

        /// Test the FIR filter stage.
        void test_filter();

        /// Test the FFT stage.
        void test_fft();

        /// Test the modules public run method.
        void test_run();

        /// Test loading a coefficients file.
        void test_loadCoeffs();

        /// Test the constructing a spectrum given a set of weights.
        void test_makeSpectrum();

        /// Test the channel profile for a given set of weights.
        void test_channelProfile();

    private:
        /// Generate configuration XML.
        QString _configXml(unsigned nChannels, unsigned nThreads = 2);

};

} // namespace lofar
} // namespace pelican

#endif // CHANNELISERPOLYPHASETEST_H
