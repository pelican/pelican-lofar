#include "test/ChanneliserPolyphaseTest.h"

#include "ChanneliserPolyphase.h"
#include "TimeStreamData.h"
#include "ChannelisedStreamData.h"
#include "PolyphaseCoefficients.h"
#include "pelican/utility/ConfigNode.h"

#include <iostream>
#include <QtCore/QTime>

#include "pelican/utility/memCheck.h"

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(ChanneliserPolyphaseTest);


void ChanneliserPolyphaseTest::setUp()
{
}

void ChanneliserPolyphaseTest::tearDown()
{
}

/**
 * @details
 * Method to test the module construction and configuration.
 */
void ChanneliserPolyphaseTest::test_configuration()
{
    unsigned nChannels = 512;

    try {
    	ConfigNode config(_configXml(nChannels));
        ChanneliserPolyphase channeliser(config);
        CPPUNIT_ASSERT_EQUAL(nChannels, channeliser._nChannels);
    }
    catch (QString err) {
        CPPUNIT_FAIL(err.toLatin1().data());
    }
}

/**
 * @details
 */
void ChanneliserPolyphaseTest::test_updateBuffer()
{
    unsigned nChan = 512;
    unsigned nSubbands = 62;
    unsigned nTaps = 8;
    unsigned nIter = 1000;

    ConfigNode config(_configXml(nChan));
    ChanneliserPolyphase channeliser(config);

    unsigned bufferSize = nChan * nTaps;
    channeliser.setupBuffers(nSubbands, nChan, nTaps);

    std::vector<std::complex<double> > sampleBuffer(nChan * nSubbands * nIter);

    std::complex<double>* subbandBuffer;
    std::complex<double>* newSamples;

    QTime timer;
    timer.start();
    for (unsigned i = 0; i < nIter; ++i) {
        for (unsigned s = 0; s < nSubbands; ++s) {
            newSamples = &sampleBuffer[i * nChan * nSubbands + s * nChan];;
            subbandBuffer = &(channeliser._subbandBuffer[s])[0];
            channeliser._updateBuffer(newSamples, nChan, subbandBuffer, bufferSize);
        }
    }
    int elapsed = timer.elapsed();
    std::cout << "Time to update buffers for " << nSubbands << " subbands = "
              << double(elapsed)/double(nIter) << " ms.\n";
}

/**
 * @details
 */
void ChanneliserPolyphaseTest::test_filter()
{
    unsigned nChan = 512;
    unsigned nSubbands = 62;
    unsigned nTaps = 8;
    ConfigNode config(_configXml(nChan));
    ChanneliserPolyphase channeliser(config);

    PolyphaseCoefficients filterCoeff(nTaps, nChan);
    const complex<double>* coeff = filterCoeff.coefficients();

    unsigned bufferSize = nChan * nTaps;
    channeliser.setupBuffers(nSubbands, nChan, nTaps);

    std::vector<complex<double> > filteredData(nChan);
    std::complex<double>* subbandBuffer;

    QTime timer;
    timer.start();
    unsigned nIter = 1000;
    for (unsigned i = 0; i < nIter; ++i) {
        for (unsigned s = 0; s < nSubbands; ++s) {
        	subbandBuffer = &(channeliser._subbandBuffer[s])[0];
            channeliser._filter(subbandBuffer, nTaps, nChan, coeff, &filteredData[0]);
        }
    }
    int elapsed = timer.elapsed();
    std::cout << "Time for ppf filter on " << nSubbands << " subbands = "
              << double(elapsed)/double(nIter) << " ms.\n";
}

/**
 * @details
 */
void ChanneliserPolyphaseTest::test_fft()
{
    unsigned nChannels = 512;
    unsigned nSubbands = 62;
    unsigned nPolarisations = 1;
    unsigned nTaps = 8;
    ConfigNode config(_configXml(nChannels));
    ChanneliserPolyphase channeliser(config);

    ChannelisedStreamData spectrum(nSubbands, nPolarisations, nChannels);
    std::vector<complex<double> > filteredData(nChannels);

    QTime timer;
    timer.start();
    unsigned nIter = 1000;
    for (unsigned i = 0; i < nIter; ++i) {
        for (unsigned s = 0; s < nSubbands; ++s) {
            channeliser._fft(&filteredData[0], nChannels, spectrum.data(s));
        }
    }
    int elapsed = timer.elapsed();
    std::cout << "Time for fft of " << nSubbands << " subbands = "
              << double(elapsed)/double(nIter) << " ms.\n";
}


/**
 * @details
 */
void ChanneliserPolyphaseTest::test_run()
{

    unsigned nChannels = 512;
    unsigned nSubbands = 62;
    unsigned nPolarisations = 1;
    unsigned nTaps = 8;
    ConfigNode config(_configXml(nChannels));
    ChanneliserPolyphase channeliser(config);
    ChannelisedStreamData spectra(nSubbands, nPolarisations, nChannels);
    PolyphaseCoefficients filterCoeff(nTaps, nChannels);
    TimeStreamData data(nSubbands, nPolarisations, nChannels);
    channeliser.setupBuffers(nSubbands, nChannels, nTaps);

    unsigned nIter = 4000;
    QTime timer;
    timer.start();
    for (unsigned i = 0; i < nIter; ++i) {
    	channeliser.run(&data, &filterCoeff, &spectra);
    }

    int elapsed = timer.elapsed();
    std::cout << "Time for run "
              << double(elapsed)/double(nIter) << " ms. " << elapsed << "\n";
}


QString ChanneliserPolyphaseTest::_configXml(const unsigned nChannels)
{
    QString xml =
            "<ChanneliserPolyphase>"
            "	<channels number=\"" + QString::number(nChannels) + "\"/>"
            "</ChanneliserPolyphase>";
    return xml;
}


} // namespace lofar
} // namespace pelican
