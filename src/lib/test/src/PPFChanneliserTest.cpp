#include "test/PPFChanneliserTest.h"

#include "PPFChanneliser.h"
#include "SpectrumDataSet.h"
#include "TimeSeriesDataSet.h"

#include "pelican/utility/ConfigNode.h"

#include <QtCore/QTime>
#include <QtCore/QFile>
#include <QtCore/QTextStream>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>

using std::cout;
using std::endl;
using std::cos;
using std::sin;
using std::vector;

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(PPFChanneliserTest);


void PPFChanneliserTest::setUp()
{
    _verbose = true;

    _nChannels = 16;
    _nSubbands = 62;
    _nPols = 2;
    _nTaps = 8;

    unsigned timesPerChunk =  512 * 1000;

    if (timesPerChunk%_nChannels) CPPUNIT_FAIL("Setup error");

    _nBlocks = timesPerChunk / _nChannels;
}



/**
 * @details
 * Test the run method.
 */
void PPFChanneliserTest::test_run()
{
    cout << endl << "***** PPFChanneliserTest::test_run() ***** " << endl;
    // Setup the channeliser.
    unsigned nThreads = 2;
    ConfigNode config(_configXml(_nChannels, nThreads, _nTaps));

    try {
        PPFChanneliser channeliser(config);
        SpectrumDataSetC32 spectra;

        // Run once to size up buffers etc.
        {
            TimeSeriesDataSetC32 timeSeries;
            timeSeries.resize(_nBlocks, _nSubbands, _nPols, _nChannels);
            channeliser.run(&timeSeries, &spectra);
        }

        TimeSeriesDataSetC32 timeSeries;
        timeSeries.resize(_nBlocks, _nSubbands, _nPols, _nChannels);

        QTime timer;
        timer.start();
        channeliser.run(&timeSeries, &spectra);
        int elapsed = timer.elapsed();

        cout << endl;
        cout << "-------------------------------------------------" << endl;
        cout << "[PPFChanneliser]: run() " << endl;
        cout << "- nChan = " << _nChannels << endl << endl;
        if (_verbose) {
            cout << "- nTaps = " << _nTaps << endl;
            cout << "- nBlocks = " << _nBlocks << endl;
            cout << "- nSubbands = " << _nSubbands << endl;
            cout << "- nPols = " << _nPols << endl;
        }
        cout << "* Elapsed = " << elapsed << " ms. [" << nThreads << " threads]";
        cout << " (data time = " << _nBlocks * _nChannels * 5e-3 << " ms.)" << endl;
        cout << "-------------------------------------------------" << endl;
    }

    catch (const QString& err)
    {
            std::cout << err.toStdString() << std::endl;
    }
    cout << endl << "***** PPFChanneliserTest::test_run() ***** " << endl;
}


/**
 * @details
 * Test to generate a channel profile.
 */
void PPFChanneliserTest::test_channelProfile()
{
    cout << endl << "*** PPFChanneliserTest::test_channelProfile() ***" << endl;

    // Setup the channeliser.
    unsigned nSubbands = 1;
    unsigned nPols = 1;
    unsigned nThreads = 1;
    unsigned nTaps = 32;
    unsigned nChannels = 64;
    ConfigNode config(_configXml(nChannels, nThreads, nTaps, "kaiser"));
    PPFChanneliser channeliser(config);
    typedef PPFChanneliser::Complex Complex;

    double sampleRate = 50.0e6; // Hz
    double startFreq = 8.0e6;   // Hz
    unsigned nSteps = 1000;     // Number of steps in profile.
    double freqInc = 0.01e6;    // Frequency increment of profile steps.
    float endFreq = startFreq + freqInc * nSteps;
    float midTestFreq = startFreq + (endFreq - startFreq) / 2.0;
    cout << "Scanning freqs " << startFreq << " -> " << endFreq << " ("
              << midTestFreq << ")" << endl;

    unsigned nProfiles = 2;
    vector<unsigned> testIndices(nProfiles);
    testIndices[0] = 45;
    testIndices[1] = 46;

    unsigned nBlocks = nTaps;

    TimeSeriesDataSetC32 data;
    data.resize(nBlocks, nSubbands, nPols, nChannels);

    SpectrumDataSetC32 spectra;
    spectra.resize(nBlocks, nSubbands, nPols, nChannels);

    vector<vector<Complex> > channelProfile;
    channelProfile.resize(nProfiles);
    for (unsigned i = 0; i < nProfiles; ++i) channelProfile[i].resize(nSteps);


    // Generate channel profile by scanning though frequencies.
    // ========================================================================
    std::vector<double> freqs(nSteps);

    for (unsigned k = 0; k < nSteps; ++k)
    {
        // Generate signal.
        freqs[k] = startFreq + k * freqInc;
        for (unsigned i = 0, t = 0; t < nBlocks; ++t)
        {
            Complex* timeData = data.timeSeriesData(t, 0, 0);
            double time, arg;
            for (unsigned c = 0; c < nChannels; ++c) {
                time = double(i++) / sampleRate;
                arg = 2.0 * math::pi * freqs[k] * time;
                timeData[c] = Complex(cos(arg), sin(arg));
            }
        }

        channeliser.run(&data, &spectra);

        // Save the amplitude of the specified channel.
        Complex* spectrum = spectra.spectrumData(nBlocks-1, 0, 0);
        for (unsigned p = 0; p < nProfiles; ++p) {
            channelProfile[p][k] = spectrum[testIndices[p]];
        }
    }


    // Write the channel profile to file.
    // ========================================================================
    QFile file("channelProfile.dat");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return;

    QTextStream out(&file);
    for (unsigned i = 0; i < nSteps; ++i) {
        out << freqs[i] << " ";
        for (unsigned p = 0; p < nProfiles; ++p)
            out << 20 * std::log10(std::abs(channelProfile[p][i])) << " ";
        out << endl;
    }
    file.close();
    cout << "*** PPFChanneliserTest::test_channelProfile() ***" << endl;
}



/**
 * @details
 * Contruct a spectrum and save it to file.
 */
void PPFChanneliserTest::test_makeSpectrum()
{
    cout << endl << "*** PPFChanneliserTest::test_makeSpectrum() ***" << endl;

    // Setup the channeliser.
    unsigned nThreads = 1;
    unsigned nSubbands = 1;
    unsigned nPols = 1;
    unsigned nChannels = 128;
    unsigned nTaps = 16;
    unsigned nBlocks = nTaps + 1;

    ConfigNode config(_configXml(nChannels, nThreads, nTaps));
    PPFChanneliser channeliser(config);
    typedef PPFChanneliser::Complex Complex;

    double freq = 12.50;      // Hz
    double sampleRate = 50.0; // Hz

    TimeSeriesDataSetC32 data;
    unsigned nTimes = nChannels;
    data.resize(nBlocks, nSubbands, nPols, nTimes);

    // Generate signal.
    for (unsigned i = 0, t = 0; t < nBlocks; ++t)
    {
        Complex* timeData = data.timeSeriesData(t, 0, 0);
        double time, arg;
        for (unsigned c = 0; c < nChannels; ++c) {
            time = double(i++) / sampleRate;
            arg = 2.0 * math::pi * freq * time;
            timeData[c] = Complex(cos(arg), sin(arg));
        }
    }

    SpectrumDataSetC32 spectra;
    spectra.resize(nBlocks, nSubbands, nPols, nChannels);

    // PPF - have to run enough times for buffer to fill with new signal.
    channeliser.run(&data, &spectra);

    // Write the last spectrum.
    QFile file("spectrum.dat");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return;

    Complex* spectrum = spectra.spectrumData(nBlocks-1, 0, 0);
    QTextStream out(&file);
    double maxFreq = sampleRate / 2.0;
    double freqInc = sampleRate / nChannels;
    for (unsigned i = 0; i < nChannels; ++i)
    {
        out << i * freqInc - maxFreq << " "
            <<  20 * std::log10(std::abs(spectrum[i])) << " "
            << std::abs(spectrum[i]) << " "
            << spectrum[i].real() << " "
            << spectrum[i].imag() << endl;
    }
    file.close();
    cout << endl << "*** PPFChanneliserTest::test_makeSpectrum() ***" << endl;
}

























/**
 * @details
 * Method to test the module construction and configuration.
 */
void PPFChanneliserTest::test_configuration()
{
    unsigned nThreads = 1;
    try {
        ConfigNode config(_configXml(_nChannels, nThreads, _nTaps));
        PPFChanneliser channeliser(config);
        CPPUNIT_ASSERT_EQUAL(_nChannels, channeliser._nChannels);
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
    unsigned nThreads = 2;
    ConfigNode config(_configXml(_nChannels, nThreads, _nTaps));
    PPFChanneliser channeliser(config);
    try {
        unsigned start = 0, end = 0;
        unsigned nSubbands = 62;
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
    catch (QString const& err) {
        CPPUNIT_FAIL(err.toLatin1().data());
    }

}


/**
 * @details
 * Test updating the delay buffering.
 */
void PPFChanneliserTest::test_updateBuffer()
{
    try {
        // Setup the channeliser.
        unsigned nThreads = 1;
        ConfigNode config(_configXml(_nChannels, nThreads, _nTaps));
        PPFChanneliser channeliser(config);

        // Setup the work buffers.
        channeliser._setupWorkBuffers(_nSubbands, _nPols, _nChannels, _nTaps);

        // Create a vector of input samples to test with.
        std::vector<PPFChanneliser::Complex> sampleBuffer(_nChannels * _nSubbands
                * _nPols * _nBlocks);

        // Local pointers to buffers.
        PPFChanneliser::Complex* subbandBuffer;
        PPFChanneliser::Complex* newSamples;

        // Iterate over the update buffer method to time it.
        QTime timer;
        timer.start();
        for (unsigned b = 0; b < _nBlocks; ++b)
        {
            for (unsigned s = 0; s < _nSubbands; ++s)
            {
                for (unsigned p = 0; p < _nPols; ++p)
                {
                    unsigned i = _nChannels * (p +  _nPols * (s + _nSubbands * b));
                    newSamples = &sampleBuffer[i];;
                    subbandBuffer = &(channeliser._workBuffer[s * _nPols + p])[0];
                    channeliser._updateBuffer(newSamples, _nChannels, _nTaps, subbandBuffer);
                }
            }
        }
        int elapsed = timer.elapsed();

        cout << endl;
        cout << "-------------------------------------------------" << endl;
        cout << "[PPFChanneliser]: _updateBuffer() " << endl;
        cout << "- nChan = " << _nChannels << endl << endl;
        if (_verbose) {
            cout << "- nTaps = " << _nTaps << endl;
            cout << "- nBlocks = " << _nBlocks << endl;
            cout << "- nSubbands = " << _nSubbands << endl;
            cout << "- nPols = " << _nPols << endl;
        }
        cout << "* Elapsed = " << elapsed << " ms. [" << nThreads << " threads]";
        cout << " (data time = " << _nBlocks * _nChannels * 5e-3 << " ms.)" << endl;
        cout << "-------------------------------------------------" << endl;
    }
    catch (QString const& err)
    {
        CPPUNIT_FAIL(err.toLatin1().data());
    }
}


/**
 * @details
 * Test the FIR filter stage.
 */
void PPFChanneliserTest::test_filter()
{
    // Setup the channeliser.
    unsigned nThreads = 1;
    ConfigNode config(_configXml(_nChannels, nThreads, _nTaps));
    PPFChanneliser channeliser(config);

    // Setup work buffers.
    channeliser._setupWorkBuffers(_nSubbands, _nPols, _nChannels, _nTaps);

    std::vector<PPFChanneliser::Complex> filteredData(_nChannels);
    PPFChanneliser::Complex* filteredSamples = &filteredData[0];
    PPFChanneliser::Complex* workBuffer;

    double const* coeff = channeliser._ppfCoeffs.ptr();
    unsigned nCoeffs =  channeliser._ppfCoeffs.size();
    float* fCoeffs = new float[nCoeffs];
    for (unsigned i = 0; i < nCoeffs; ++i) fCoeffs[i] = float(coeff[i]);

    QTime timer;
    timer.start();
    for (unsigned b = 0; b < _nBlocks; ++b) {
        for (unsigned s = 0; s < _nSubbands; ++s) {
            for (unsigned p = 0; p < _nPols; ++p) {
                workBuffer = &(channeliser._workBuffer[s * _nPols + p])[0];
                channeliser._filter(workBuffer, _nTaps, _nChannels, fCoeffs,
                        filteredSamples);
            }
        }
    }
    int elapsed = timer.elapsed();

    cout << endl;
    cout << "-------------------------------------------------" << endl;
    cout << "[PPFChanneliser]: _filter() " << endl;
    cout << "- nChan = " << _nChannels << endl << endl;
    if (_verbose) {
        cout << "- nTaps = " << _nTaps << endl;
        cout << "- nBlocks = " << _nBlocks << endl;
        cout << "- nSubbands = " << _nSubbands << endl;
        cout << "- nPols = " << _nPols << endl;
    }
    cout << "* Elapsed = " << elapsed << " ms. [" << nThreads << " threads]";
    cout << " (data time = " << _nBlocks * _nChannels * 5e-3 << " ms.)" << endl;
    cout << "-------------------------------------------------" << endl;
}


/**
 * @details
 * Test the fft stage.
 */
void PPFChanneliserTest::test_fft()
{
    // Setup the channeliser.
    unsigned nThreads = 1;
    ConfigNode config(_configXml(_nChannels, nThreads, _nTaps));
    PPFChanneliser channeliser(config);

    SpectrumDataSetC32 spectra;
    spectra.resize(_nBlocks, _nSubbands, _nPols);
    std::vector<PPFChanneliser::Complex> filteredData(_nChannels);
    const PPFChanneliser::Complex* filteredSamples = &filteredData[0];

    QTime timer;
    timer.start();
    for (unsigned i = 0, b = 0; b < _nBlocks; ++b) {
        for (unsigned s = 0; s < _nSubbands; ++s) {
            for (unsigned p = 0; p < _nPols; ++p) {

                Spectrum<PPFChanneliser::Complex>* spectrum = spectra.spectrum(i++);
                spectrum->resize(_nChannels);
                PPFChanneliser::Complex* spectrumData = spectrum->data();
                channeliser._fft(filteredSamples, spectrumData);
            }
        }
    }
    int elapsed = timer.elapsed();

    cout << endl;
    cout << "-------------------------------------------------" << endl;
    cout << "[PPFChanneliser]: _fft() " << endl;
    cout << "- nChan = " << _nChannels << endl << endl;
    if (_verbose) {
        cout << "- nTaps = " << _nTaps << endl;
        cout << "- nBlocks = " << _nBlocks << endl;
        cout << "- nSubbands = " << _nSubbands << endl;
        cout << "- nPols = " << _nPols << endl;
    }
    cout << "* Elapsed = " << elapsed << " ms. [" << nThreads << " threads]";
    cout << " (data time = " << _nBlocks * _nChannels * 5e-3 << " ms.)" << endl;
    cout << "-------------------------------------------------" << endl;
}







/**
 * @details
 *
 * @param nChannels
 * @return
 */
QString PPFChanneliserTest::_configXml(unsigned nChannels,
        unsigned nThreads, unsigned nTaps, const QString& windowType)
{
    QString xml =
            "<PPFChanneliser>"
            "	<outputChannelsPerSubband value=\"" + QString::number(nChannels) + "\"/>"
            "	<processingThreads value=\"" + QString::number(nThreads) + "\"/>"
            "	<filter nTaps=\"" + QString::number(nTaps) + "\" filterWindow=\"" + windowType + "\"/>"
            "</PPFChanneliser>";
    return xml;
}


} // namespace lofar
} // namespace pelican
