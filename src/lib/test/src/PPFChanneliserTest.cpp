#include "test/PPFChanneliserTest.h"

#include "PPFChanneliser.h"
#include "SubbandSpectra.h"
#include "SubbandTimeSeries.h"

#include "pelican/utility/ConfigNode.h"

#include <QtCore/QTime>
#include <QtCore/QFile>
#include <QtCore/QTextStream>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>

#include "pelican/utility/memCheck.h"

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(PPFChanneliserTest);


/**
 * @details
 * Method to test the module construction and configuration.
 */
void PPFChanneliserTest::test_configuration()
{
    unsigned nChannels = 512;
    unsigned nThreads = 3;
    QString coeffFile = "lib/test/data/coeffs_512_1.dat";
    unsigned nTaps = 8;
    if (!QFile::exists(coeffFile)) return;
    try {
        ConfigNode config(_configXml(nChannels, nThreads, nTaps, coeffFile));
        PPFChanneliser channeliser(config);
        CPPUNIT_ASSERT_EQUAL(nChannels, channeliser._nChannels);
        CPPUNIT_ASSERT_EQUAL(nThreads, channeliser._nThreads);
    }
    catch (QString err) {
        CPPUNIT_FAIL(err.toLatin1().data());
    }
}


/**
 * @details
 */
void PPFChanneliserTest::test_threadAssign()
{
    unsigned nChannels = 512;
    unsigned nThreads = 2;
    QString coeffFile = "lib/test/data/coeffs_512_1.dat";
    if (!QFile::exists(coeffFile)) return;
    unsigned nTaps = 8;
    ConfigNode config(_configXml(nChannels, nThreads, nTaps, coeffFile));
    PPFChanneliser channeliser(config);


    unsigned nSubbands = 62;
    unsigned start = 0, end = 0;

    try {
        unsigned threadId = 0;
        channeliser._threadProcessingIndices(start, end, nSubbands, nThreads, threadId);
        CPPUNIT_ASSERT_EQUAL(unsigned(0), start);
        CPPUNIT_ASSERT_EQUAL(unsigned(31), end);

        threadId = 1;
        channeliser._threadProcessingIndices(start, end, nSubbands, nThreads, threadId);
        CPPUNIT_ASSERT_EQUAL(unsigned(31), start);
        CPPUNIT_ASSERT_EQUAL(unsigned(62), end);

        nSubbands = 1;
        threadId = 0;
        channeliser._threadProcessingIndices(start, end, nSubbands, nThreads, threadId);
        CPPUNIT_ASSERT_EQUAL(unsigned(0), start);
        CPPUNIT_ASSERT_EQUAL(unsigned(1), end);

        threadId = 1;
        channeliser._threadProcessingIndices(start, end, nSubbands, nThreads, threadId);
        CPPUNIT_ASSERT_EQUAL(unsigned(0), start);
        CPPUNIT_ASSERT_EQUAL(unsigned(0), end);

        nSubbands = 3;
        threadId = 0;
        channeliser._threadProcessingIndices(start, end, nSubbands, nThreads, threadId);
        CPPUNIT_ASSERT_EQUAL(unsigned(0), start);
        CPPUNIT_ASSERT_EQUAL(unsigned(2), end);

        threadId = 1;
        channeliser._threadProcessingIndices(start, end, nSubbands, nThreads, threadId);
        CPPUNIT_ASSERT_EQUAL(unsigned(2), start);
        CPPUNIT_ASSERT_EQUAL(unsigned(3), end);

        nSubbands = 4;
        threadId = 0;
        channeliser._threadProcessingIndices(start, end, nSubbands, nThreads, threadId);
        CPPUNIT_ASSERT_EQUAL(unsigned(0), start);
        CPPUNIT_ASSERT_EQUAL(unsigned(2), end);

        threadId = 1;
        channeliser._threadProcessingIndices(start, end, nSubbands, nThreads, threadId);
        CPPUNIT_ASSERT_EQUAL(unsigned(2), start);
        CPPUNIT_ASSERT_EQUAL(unsigned(4), end);

        nSubbands = 4;
        nThreads = 3;
        threadId = 0;
        channeliser._threadProcessingIndices(start, end, nSubbands, nThreads, threadId);
        CPPUNIT_ASSERT_EQUAL(unsigned(0), start);
        CPPUNIT_ASSERT_EQUAL(unsigned(2), end);

        threadId = 1;
        channeliser._threadProcessingIndices(start, end, nSubbands, nThreads, threadId);
        CPPUNIT_ASSERT_EQUAL(unsigned(2), start);
        CPPUNIT_ASSERT_EQUAL(unsigned(3), end);

        threadId = 2;
        channeliser._threadProcessingIndices(start, end, nSubbands, nThreads, threadId);
        CPPUNIT_ASSERT_EQUAL(unsigned(3), start);
        CPPUNIT_ASSERT_EQUAL(unsigned(4), end);
    }
    catch (QString err) {
        CPPUNIT_FAIL(err.toLatin1().data());
    }

}


/**
 * @details
 * Test updating the delay buffering.
 */
void PPFChanneliserTest::test_updateBuffer()
{
    // Setup the channeliser.
    unsigned nChan = 512;
    unsigned nThreads = 1;
    QString coeffFile = "lib/test/data/coeffs_512_1.dat";
    if (!QFile::exists(coeffFile)) return;
    unsigned nTaps = 8;
    ConfigNode config(_configXml(nChan, nThreads, nTaps, coeffFile));
    PPFChanneliser channeliser(config);

    // Setup the work buffers.
    unsigned nPol = 2;
    unsigned nSubbands = 62;
    unsigned bufferSize = nChan * nTaps;
    channeliser._setupWorkBuffers(nSubbands, nPol, nChan, nTaps);

    // Create a vector of input samples to test with.
    unsigned nIter = 500;
    std::vector<PPFChanneliser::Complex> sampleBuffer(nChan * nSubbands
            * nPol * nIter);

    // Local pointers to buffers.
    PPFChanneliser::Complex* subbandBuffer;
    PPFChanneliser::Complex* newSamples;

    // Iterate over the update buffer method to time it.
    QTime timer;
    timer.start();
    for (unsigned i = 0; i < nIter; ++i) {

        for (unsigned s = 0; s < nSubbands; ++s) {
            for (unsigned p = 0; p < nPol; ++p) {
                unsigned index = i * nChan * nSubbands * nPol +
                        s * nChan * nPol + p * nChan;
                newSamples = &sampleBuffer[index];;
                subbandBuffer = &(channeliser._workBuffer[s * nPol + p])[0];
                channeliser._updateBuffer(newSamples, nChan, subbandBuffer,
                        bufferSize);
            }
        }

    }
    int elapsed = timer.elapsed();
    std::cout << "\n[PPFChanneliser]: Time to update buffers for "
              << nSubbands << " subbands, " << nPol << " polarisations = "
              << double(elapsed)/double(nIter) << " ms. (1 thread)\n";
}


/**
 * @details
 * Test the FIR filter stage.
 */
void PPFChanneliserTest::test_filter()
{
    // Setup the channeliser.
    unsigned nChan = 512;
    unsigned nThreads = 1;
    QString coeffFile = "lib/test/data/coeffs_512_1.dat";
    if (!QFile::exists(coeffFile))
        return;
    unsigned nTaps = 8;
    ConfigNode config(_configXml(nChan, nThreads, nTaps, coeffFile));
    PPFChanneliser channeliser(config);

    // Setup work buffers.
    unsigned bufferSize = nChan * nTaps;
    unsigned nSubbands = 62;
    unsigned nPol = 2;
    unsigned nTimeBlocks = 1;
    channeliser._setupWorkBuffers(nSubbands, nPol, nChan, nTaps);

    std::vector<PPFChanneliser::Complex> filteredData(nChan);
    PPFChanneliser::Complex* filteredSamples = &filteredData[0];
    PPFChanneliser::Complex* workBuffer;

    const double* coeff = channeliser._coeffs.ptr();

    QTime timer;
    timer.start();
    unsigned nIter = 500;
    for (unsigned b = 0; b < nTimeBlocks; ++b) {
        for (unsigned i = 0; i < nIter; ++i) {
            for (unsigned s = 0; s < nSubbands; ++s) {
                for (unsigned p = 0; p < nPol; ++p) {
                    workBuffer = &(channeliser._workBuffer[s * nPol + p])[0];
                    channeliser._filter(workBuffer, nTaps, nChan, coeff,
                            filteredSamples);
                }
            }
        }
    }
    int elapsed = timer.elapsed();
    std::cout << "\n[PPFChanneliser]: Time for PPF filtering of "
            << nSubbands << " subbands, " << nPol << " polarisations = "
            << double(elapsed)/double(nIter) << " ms. (1 thread)\n";
}


/**
 * @details
 * Test the fft stage.
 */
void PPFChanneliserTest::test_fft()
{
    // Setup the channeliser.
    unsigned nChan = 512;
    unsigned nThreads = 1;
    QString coeffFile = "lib/test/data/coeffs_512_1.dat";
    if (!QFile::exists(coeffFile)) return;
    unsigned nTaps = 8;
    ConfigNode config(_configXml(nChan, nThreads, nTaps, coeffFile));
    PPFChanneliser channeliser(config);

    unsigned nSubbands = 62;
    unsigned nPol = 2;
    unsigned nTimeBlocks = 1;
    SubbandSpectraC32 spectra;
    spectra.resize(nTimeBlocks, nSubbands, nPol);
    std::vector<PPFChanneliser::Complex> filteredData(nChan);
    const PPFChanneliser::Complex* filteredSamples = &filteredData[0];

    QTime timer;
    timer.start();
    unsigned nIter = 1000;
    for (unsigned i = 0; i < nIter; ++i) {
        for (unsigned b = 0; b < nTimeBlocks; ++b) {
            for (unsigned s = 0; s < nSubbands; ++s) {
                for (unsigned p = 0; p < nPol; ++p) {
                    Spectrum<PPFChanneliser::Complex>* spectrum = spectra.ptr(b, s, p);
                    spectrum->resize(nChan);
                    PPFChanneliser::Complex* spectrumData = spectrum->ptr();
                    channeliser._fft(filteredSamples, nChan, spectrumData);
                }
            }
        }
    }
    int elapsed = timer.elapsed();
    std::cout << "\n[PPFChanneliser]: Time for the FFT of "
            << nSubbands << " subbands, " << nPol << " polarisations, "
            << nTimeBlocks << " time blocks = "
            << double(elapsed)/double(nIter) << " ms. (1 thread)\n";
}


/**
 * @details
 * Test the run method.
 */
void PPFChanneliserTest::test_run()
{
    // Setup the channeliser.
    unsigned nChan = 512;
    unsigned nThreads = 2;
    QString coeffFile = "lib/test/data/coeffs_512_1.dat";
    if (!QFile::exists(coeffFile)) return;
    unsigned nTaps = 8;
    ConfigNode config(_configXml(nChan, nThreads, nTaps, coeffFile));
    PPFChanneliser channeliser(config);

    unsigned nSubbands = 62;
    unsigned nTimeBlocks = 1;
    unsigned nPol = 2;
    SubbandSpectraC32 spectra;
    spectra.resize(nTimeBlocks, nSubbands, nPol);
    SubbandTimeSeriesC32 timeSeries;
    timeSeries.resize(nTimeBlocks, nSubbands, nPol);
    for (unsigned i = 0; i < timeSeries.nTimeSeries(); ++i) {
        TimeSeries<PPFChanneliser::Complex>* t = timeSeries.ptr(i);
        t->resize(nChan);
    }

    unsigned nIter = 100;
    QTime timer;
    timer.start();
    for (unsigned i = 0; i < nIter; ++i) {
        channeliser.run(&timeSeries, &spectra);
    }
    int elapsed = timer.elapsed();
    std::cout << "\n[PPFChanneliser]: Time for run method using "
            << nSubbands << " subbands, " << nPol << " polarisations, "
            << nTimeBlocks << " time blocks = "
            << double(elapsed)/double(nIter) << " ms. ("
            << nThreads << " threads)\n";
}



/**
 * @details
 * Contruct a spectrum and save it to file.
 */
void PPFChanneliserTest::test_makeSpectrum()
{
    // Setup the channeliser.
    unsigned nChan = 64;
    unsigned nThreads = 1;
    QString coeffFile = "lib/test/data/coeffs_64_1.dat";
    if (!QFile::exists(coeffFile)) return;
    unsigned nTaps = 8;
    ConfigNode config(_configXml(nChan, nThreads, nTaps, coeffFile));
    PPFChanneliser channeliser(config);

    unsigned nSubbands = 1;
    unsigned nPol = 1;

    unsigned nTimeBlocks = 1000;
    double freq = 10.12; // Hz
    double sampleRate = 50.0; // Hz


    SubbandTimeSeriesC32 data;
    data.resize(nTimeBlocks, nSubbands, nPol, nChan);

    // Generate signal.
    for (unsigned i = 0, t = 0; t < nTimeBlocks; ++t) {
        PPFChanneliser::Complex* timeData = data.ptr(t)->ptr();
        for (unsigned c = 0; c < nChan; ++c) {
            double t = double(i) / sampleRate;
            double re = std::cos(2 * math::pi * freq * t);
            double im = std::sin(2 * math::pi * freq * t);
            timeData[c] = PPFChanneliser::Complex(re, im);
            i++;
        }
    }

    SubbandSpectraC32 spectra;
    spectra.resize(nTimeBlocks, nSubbands, nPol, nChan);

    // PPF - have to run enough times for buffer to fill with new signal.
    channeliser.run(&data, &spectra);

    // Write the last spectrum.
    QFile file("spectrum.dat");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return;
    }

    PPFChanneliser::Complex* spectrum = spectra.ptr(nTimeBlocks-1, 0, 0)->ptr();
    QTextStream out(&file);
    double maxFreq = sampleRate / 2.0;
    double freqInc = sampleRate / nChan;
    for (unsigned i = 0; i < nChan; ++i) {
        out << i * freqInc - maxFreq << " "
            <<	20 * std::log10(std::abs(spectrum[i])) << " "
            << std::abs(spectrum[i]) << " "
            << spectrum[i].real() << " "
            << spectrum[i].imag() << endl;
    }
    file.close();
}




/**
 * @details
 * Test to generate a channel profile.
 */
void PPFChanneliserTest::test_channelProfile()
{
    // Options.
    //--------------------------------------------------------------------------
    unsigned nChannels = 64;
    unsigned nSubbands = 1;
    unsigned nPolarisations = 1;
    unsigned nSamples = nChannels;
    unsigned nTaps = 8;
    unsigned nThreads = 1;
    QString coeffFile = "lib/test/data/coeffs_64_1.dat";
    if (!QFile::exists(coeffFile)) return;

    unsigned nProfiles = 2;
    double sampleRate = 50.0; // Hz
    double startFreq = 8.0; // Hz
    unsigned nSteps = 1000;
    double freqInc = 0.01;
    std::vector<double> freqs(nSteps);

    double channelDelta = double(nChannels) / double(sampleRate);
    double endFreq = startFreq + freqInc * nSteps;
    double midTestFreq = startFreq + (endFreq - startFreq) / 2.0;
    std::cout << "scanning freqs " << startFreq << " -> " << endFreq << " ("
              << midTestFreq << ")" << std::endl;

    //	unsigned testChannelIndex = nChannels / 2 + std::floor(midTestFreq / channelDelta);
    std::vector<unsigned> testIndices(nProfiles);
    testIndices[0] = 45;
    testIndices[1] = 46;

    try {
        ConfigNode config(_configXml(nChannels, nThreads, nTaps, coeffFile));
        PPFChanneliser channeliser(config);

        unsigned nTimeBlocks = nTaps;

        SubbandTimeSeriesC32 data;
        data.resize(nTimeBlocks, nSubbands, nPolarisations, nChannels);

        SubbandSpectraC32 spectra;
        spectra.resize(nTimeBlocks, nSubbands, nPolarisations, nChannels);

        std::vector<std::vector<PPFChanneliser::Complex> > channelProfile;
        channelProfile.resize(nProfiles);
        for (unsigned i = 0; i < nProfiles; ++i) {
            channelProfile[i].resize(nSteps);
        }

        // Scan frequencies to generate channel profile.
        for (unsigned k = 0; k < nSteps; ++k) {

            // Generate signal.
            double freq = startFreq + k * freqInc;
            freqs[k] = freq;
            for (unsigned i = 0, t = 0; t < nTimeBlocks; ++t) {
                PPFChanneliser::Complex* timeData = data.ptr(t)->ptr();
                for (unsigned c = 0; c < nChannels; ++c) {
                    double t = double(i) / sampleRate;
                    double re = std::cos(2 * math::pi * freq * t);
                    double im = std::sin(2 * math::pi * freq * t);
                    timeData[c] = PPFChanneliser::Complex(re, im);
                    i++;
                }
            }

            channeliser.run(&data, &spectra);

            // Save the amplitude of the specified channel.
            PPFChanneliser::Complex* spectrum =
                    spectra.ptr(nTimeBlocks-1, 0, 0)->ptr();
            for (unsigned p = 0; p < nProfiles; ++p) {
                channelProfile[p][k] = spectrum[testIndices[p]];
            }
        }

        // Write the channel profile to file.
        QFile file("channelProfile.dat");
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            return;
        }
        QTextStream out(&file);
        for (unsigned i = 0; i < nSteps; ++i) {
            out << freqs[i] << " ";
            for (unsigned p = 0; p < nProfiles; ++p) {
                out << 20 * std::log10(std::abs(channelProfile[p][i])) << " ";
            }
            out << endl;
        }
        file.close();
    }
    catch (QString err) {
        std::cout << err.toStdString() << std::endl;
    }
}



/**
 * @details
 * Test to generate a channel profile.
 */
void PPFChanneliserTest::test_channelProfileGeneratedWeights()
{
    // Options.
    //--------------------------------------------------------------------------
    unsigned nChannels = 64;
    unsigned nSubbands = 1;
    unsigned nPolarisations = 1;
    unsigned nSamples = nChannels;
    unsigned nTaps = 8;
    unsigned nThreads = 1;
    QString coeffFile = "";

    unsigned nProfiles = 2;
    double sampleRate = 50.0e6; // Hz
    double startFreq = 8.0e6; // Hz
    unsigned nSteps = 1000;
    double freqInc = 0.01e6;
    std::vector<double> freqs(nSteps);

    double channelDelta = double(nChannels) / double(sampleRate);
    double endFreq = startFreq + freqInc * nSteps;
    double midTestFreq = startFreq + (endFreq - startFreq) / 2.0;
    std::cout << "scanning freqs " << startFreq << " -> " << endFreq << " ("
              << midTestFreq << ")" << std::endl;

    //	unsigned testChannelIndex = nChannels / 2 + std::floor(midTestFreq / channelDelta);
    std::vector<unsigned> testIndices(nProfiles);
    testIndices[0] = 45;
    testIndices[1] = 46;

    try {
        ConfigNode config(_configXml(nChannels, nThreads, nTaps, coeffFile,
                "kaiser"));
        PPFChanneliser channeliser(config);

        unsigned nTimeBlocks = nTaps;

        SubbandTimeSeriesC32 data;
        data.resize(nTimeBlocks, nSubbands, nPolarisations, nChannels);

        SubbandSpectraC32 spectra;
        spectra.resize(nTimeBlocks, nSubbands, nPolarisations, nChannels);

        std::vector<std::vector<PPFChanneliser::Complex> > channelProfile;
        channelProfile.resize(nProfiles);
        for (unsigned i = 0; i < nProfiles; ++i) {
            channelProfile[i].resize(nSteps);
        }

        // Scan frequencies to generate channel profile.
        for (unsigned k = 0; k < nSteps; ++k) {

            // Generate signal.
            double freq = startFreq + k * freqInc;
            freqs[k] = freq;
            for (unsigned i = 0, t = 0; t < nTimeBlocks; ++t) {
                PPFChanneliser::Complex* timeData = data.ptr(t)->ptr();
                for (unsigned c = 0; c < nChannels; ++c) {
                    double t = double(i) / sampleRate;
                    double re = std::cos(2 * math::pi * freq * t);
                    double im = std::sin(2 * math::pi * freq * t);
                    timeData[c] = PPFChanneliser::Complex(re, im);
                    i++;
                }
            }

            channeliser.run(&data, &spectra);

            // Save the amplitude of the specified channel.
            PPFChanneliser::Complex* spectrum =
                    spectra.ptr(nTimeBlocks-1, 0, 0)->ptr();
            for (unsigned p = 0; p < nProfiles; ++p) {
                channelProfile[p][k] = spectrum[testIndices[p]];
            }
        }

        // Write the channel profile to file.
        QFile file("channelProfileGeneratedCoeffs.dat");
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            return;
        }
        QTextStream out(&file);
        for (unsigned i = 0; i < nSteps; ++i) {
            out << freqs[i] << " ";
            for (unsigned p = 0; p < nProfiles; ++p) {
                out << 20 * std::log10(std::abs(channelProfile[p][i])) << " ";
            }
            out << endl;
        }
        file.close();
    }
    catch (QString err) {
        std::cout << err.toStdString() << std::endl;
    }
}



/**
 * @details
 *
 * @param nChannels
 * @return
 */
QString PPFChanneliserTest::_configXml(unsigned nChannels,
        unsigned nThreads, unsigned nTaps, const QString& coeffFile,
        const QString& windowType)
{
    QString xml =
            "<PPFChanneliser>"
            "	<channels number=\"" + QString::number(nChannels) + "\"/>"
            "	<processingThreads number=\"" + QString::number(nThreads) + "\"/>"
            "	<filter fileName=\"" + coeffFile + "\" nTaps=\"" +
                QString::number(nTaps) + "\" filterWindow=\"" +
                    windowType + "\"/>"
            "</PPFChanneliser>";
    return xml;
}


} // namespace lofar
} // namespace pelican
