#include "SpectrumTest.h"
#include "Spectrum.h"

#include <QtCore/QBuffer>

#include <iostream>
#include <complex>

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(SpectrumTest);


/**
 * @details
 * Tests the various accessor methods for the time stream data blob
 */
void SpectrumTest::test_accessorMethods()
{
    // Use Case
    // Construct a spectrum data blob and test each of the accessor methods.
    {
        // Error tolerance use for double comparisons.
        double err = 1.0e-5;

        // Check default constructor
        {
            Spectrum<float> spectrum;
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, spectrum.startFrequency(), err);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, spectrum.channelFrequencyDelta(),
                    err);
            CPPUNIT_ASSERT_EQUAL(unsigned(0), spectrum.nChannels());
        }

        unsigned nChan = 20;
        Spectrum<float> spectrum(nChan);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, spectrum.startFrequency(), err);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, spectrum.channelFrequencyDelta(), err);
        CPPUNIT_ASSERT_EQUAL(nChan, spectrum.nChannels());

        nChan = 256;
        spectrum.resize(nChan);
        CPPUNIT_ASSERT_EQUAL(nChan, spectrum.nChannels());

        double startFreq = 1.2345e6;
        spectrum.setStartFrequency(startFreq);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(startFreq, spectrum.startFrequency(), err);

        double freqDelta = 0.987e8;
        spectrum.setChannelFrequencyDelta(freqDelta);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(freqDelta,
                spectrum.channelFrequencyDelta(), err);

        nChan = 5;
        spectrum.resize(nChan);
        CPPUNIT_ASSERT_EQUAL(nChan, spectrum.nChannels());
        float* sIn = spectrum.ptr();
        for (unsigned i = 0; i < nChan; ++i) {
            sIn[i] = float(i) * 1.1;
        }
        const float* sOut = spectrum.ptr();
        for (unsigned i = 0; i < nChan; ++i) {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(float(i) * 1.1, sOut[i], err);
        }
    }
}

} // namespace lofar
} // namespace pelican
