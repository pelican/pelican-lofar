#include "test/ChanneliserPolyphaseTest.h"

#include "ChanneliserPolyphase.h"
#include "TimeStreamData.h"
#include "ChannelisedStreamData.h"
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
    unsigned nTaps = 8;
    unsigned nSubbands = 62;
    QString fileName = "coeffs.dat";
    ConfigNode config;
    _setupConfig(config, nChannels, nTaps, nSubbands, fileName);

    try {
        ChanneliserPolyphase channeliser(config);
        CPPUNIT_ASSERT_EQUAL(nChannels, channeliser._nChannels);
        CPPUNIT_ASSERT_EQUAL(nChannels, channeliser._nChannels);
        CPPUNIT_ASSERT_EQUAL(nTaps, channeliser._nFilterTaps);
        CPPUNIT_ASSERT_EQUAL(nSubbands, channeliser._nSubbands);
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
    ConfigNode config;
    _setupConfig(config, nChan, 8, 62, "coeffs.dat");
    ChanneliserPolyphase channeliser(config);
    unsigned bufferSize = channeliser._subbandBuffer[0].size();

    std::complex<double>* sampleBuffer;

    QTime timer;
    timer.start();
    unsigned nIter = 1000;
    for (unsigned i = 0; i < nIter; ++i) {
        for (unsigned s = 0; s < nSubbands; ++s) {
            std::vector<std::complex<double> > newSamples(nChan);
            sampleBuffer = &(channeliser._subbandBuffer[s])[0];
            channeliser._updateBuffer(&newSamples[0], nChan, sampleBuffer, bufferSize);
        }
    }
    int elapsed = timer.elapsed();
    std::cout << "Elapsed time = " << elapsed / 1.0e3 << " s.\n";
    std::cout << "Time per spectra = "
              << double(elapsed)/(double(nIter) * double(nSubbands)) << " ms.\n";
}

/**
 * @details
 */
void ChanneliserPolyphaseTest::test_filter()
{
    unsigned nChan = 512;
    unsigned nSubbands = 62;
    unsigned nTaps = 8;
    ConfigNode config;
    _setupConfig(config, nChan, nTaps, 62, "coeffs.dat");
    ChanneliserPolyphase channeliser(config);
    unsigned bufferSize = channeliser._subbandBuffer[0].size();

    std::complex<double>* sampleBuffer;
    //std::cout << "nCoeffs = " << channeliser._filterCoeff.size() << std::endl;
    //std::cout << "bufferSize = " << bufferSize << std::endl;

    QTime timer;
    timer.start();
    unsigned nIter = 100;
    for (unsigned i = 0; i < nIter; ++i) {
        for (unsigned s = 0; s < nSubbands; ++s) {
            sampleBuffer = &(channeliser._subbandBuffer[s])[0];
            std::vector<std::complex<double> > filteredBuffer(nChan);
            const complex<double>* coeff = channeliser._filterCoeff.coefficients();
            channeliser._filter(sampleBuffer, nTaps, nChan, coeff, &filteredBuffer[0]);
        }
    }
    int elapsed = timer.elapsed();
    std::cout << "Elapsed time = " << elapsed / 1.0e3 << " s.\n";
    std::cout << "Time per filter = "
              << double(elapsed)/(double(nIter) * double(nSubbands)) << " ms.\n";
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
    ConfigNode config;
    _setupConfig(config, nChannels, nTaps, 62, "coeffs.dat");

    ChanneliserPolyphase channeliser(config);
    ChannelisedStreamData spectrum(nSubbands, nPolarisations, nChannels);
    std::complex<double>* subbandSpectrum;

    QTime timer;
    timer.start();
    unsigned nIter = 1000;
    for (unsigned i = 0; i < nIter; ++i) {
        for (unsigned s = 0; s < nSubbands; ++s) {
        	std::vector<std::complex<double> > filteredBuffer(nChannels, std::complex<double>(0.0, 0.0));
        	subbandSpectrum = spectrum.data(s);
            channeliser._fft(&filteredBuffer[0], nChannels, subbandSpectrum);
        }
    }
    int elapsed = timer.elapsed();
    std::cout << "Elapsed time = " << elapsed / 1.0e3 << " s.\n";
    std::cout << "Time per fft = "
              << double(elapsed)/(double(nIter) * double(nSubbands)) << " ms.\n";
}


void ChanneliserPolyphaseTest::_setupConfig(ConfigNode& config,
        const unsigned nChannels, const unsigned nTaps,
        const unsigned nSubbands, const QString coeffFile)
{
    QString xml =
            "<ChanneliserPolyphase>"
            "	<channels number=\"" + QString::number(nChannels) + "\"/>"
            "	<filterTaps number=\"" + QString::number(nTaps) + "\"/>"
            "	<subbands number=\"" + QString::number(nSubbands) + "\"/>"
            "   <coefficients fileName=\"" + coeffFile + "\"/>"
            "</ChanneliserPolyphase>";
    try {
        config.setFromString(xml);
    }
    catch (QString err) {
        std::cout << err.toStdString() << std::endl;
    }
}


} // namespace lofar
} // namespace pelican
