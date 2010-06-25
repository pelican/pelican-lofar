#include "test/ChanneliserPolyphaseTest.h"

#include "ChanneliserPolyphase.h"
#include "TimeStreamData.h"
#include "ChannelisedStreamData.h"
#include "PolyphaseCoefficients.h"
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

CPPUNIT_TEST_SUITE_REGISTRATION(ChanneliserPolyphaseTest);


/**
 * @details
 * Method to test the module construction and configuration.
 */
void ChanneliserPolyphaseTest::test_configuration()
{
    unsigned nChannels = 512;
    unsigned nThreads = 3;
    try {
    	ConfigNode config(_configXml(nChannels, nThreads));
        ChanneliserPolyphase channeliser(config);
        CPPUNIT_ASSERT_EQUAL(nChannels, channeliser._nChannels);
        CPPUNIT_ASSERT_EQUAL(nThreads, channeliser._nThreads);
    }
    catch (QString err) {
        CPPUNIT_FAIL(err.toLatin1().data());
    }
}


void ChanneliserPolyphaseTest::test_threadAssign()
{
	unsigned nThreads = 2;
	unsigned nSubbands = 62;
	unsigned start = 0, end = 0;

	ConfigNode config(_configXml(nSubbands));
	ChanneliserPolyphase channeliser(config);

	try {
		unsigned threadId = 0;
		channeliser._threadSubbandRange(start, end, nSubbands, nThreads, threadId);
		CPPUNIT_ASSERT_EQUAL(unsigned(0), start);
		CPPUNIT_ASSERT_EQUAL(unsigned(31), end);

		threadId = 1;
		channeliser._threadSubbandRange(start, end, nSubbands, nThreads, threadId);
		CPPUNIT_ASSERT_EQUAL(unsigned(31), start);
		CPPUNIT_ASSERT_EQUAL(unsigned(62), end);

		nSubbands = 1;
		threadId = 0;
		channeliser._threadSubbandRange(start, end, nSubbands, nThreads, threadId);
		CPPUNIT_ASSERT_EQUAL(unsigned(0), start);
		CPPUNIT_ASSERT_EQUAL(unsigned(1), end);

		threadId = 1;
		channeliser._threadSubbandRange(start, end, nSubbands, nThreads, threadId);
		CPPUNIT_ASSERT_EQUAL(unsigned(0), start);
		CPPUNIT_ASSERT_EQUAL(unsigned(0), end);

		nSubbands = 3;
		threadId = 0;
		channeliser._threadSubbandRange(start, end, nSubbands, nThreads, threadId);
		CPPUNIT_ASSERT_EQUAL(unsigned(0), start);
		CPPUNIT_ASSERT_EQUAL(unsigned(2), end);

		threadId = 1;
		channeliser._threadSubbandRange(start, end, nSubbands, nThreads, threadId);
		CPPUNIT_ASSERT_EQUAL(unsigned(2), start);
		CPPUNIT_ASSERT_EQUAL(unsigned(3), end);

		nSubbands = 4;
		threadId = 0;
		channeliser._threadSubbandRange(start, end, nSubbands, nThreads, threadId);
		CPPUNIT_ASSERT_EQUAL(unsigned(0), start);
		CPPUNIT_ASSERT_EQUAL(unsigned(2), end);

		threadId = 1;
		channeliser._threadSubbandRange(start, end, nSubbands, nThreads, threadId);
		CPPUNIT_ASSERT_EQUAL(unsigned(2), start);
		CPPUNIT_ASSERT_EQUAL(unsigned(4), end);

		nSubbands = 4;
		nThreads = 3;
		threadId = 0;
		channeliser._threadSubbandRange(start, end, nSubbands, nThreads, threadId);
		CPPUNIT_ASSERT_EQUAL(unsigned(0), start);
		CPPUNIT_ASSERT_EQUAL(unsigned(2), end);

		threadId = 1;
		channeliser._threadSubbandRange(start, end, nSubbands, nThreads, threadId);
		CPPUNIT_ASSERT_EQUAL(unsigned(2), start);
		CPPUNIT_ASSERT_EQUAL(unsigned(3), end);

		threadId = 2;
		channeliser._threadSubbandRange(start, end, nSubbands, nThreads, threadId);
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
void ChanneliserPolyphaseTest::test_updateBuffer()
{
    unsigned nChan = 512;
    unsigned nSubbands = 62;
    unsigned nTaps = 8;
    unsigned nIter = 1000;

    ConfigNode config(_configXml(nChan));
    ChanneliserPolyphase channeliser(config);

    unsigned bufferSize = nChan * nTaps;
    channeliser._setupBuffers(nSubbands, nChan, nTaps);

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
 * Test the FIR filter stage.
 */
void ChanneliserPolyphaseTest::test_filter()
{
    unsigned nChan = 512;
    unsigned nSubbands = 62;
    unsigned nTaps = 8;

    ConfigNode config(_configXml(nChan));
    ChanneliserPolyphase channeliser(config);

    PolyphaseCoefficients filterCoeff(nTaps, nChan);
    const double* coeff = filterCoeff.coefficients();

    unsigned bufferSize = nChan * nTaps;
    channeliser._setupBuffers(nSubbands, nChan, nTaps);

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
    std::cout << "Time taken for PPF filtering of " << nSubbands
    		  << " subbands = " << double(elapsed)/double(nIter) << " ms.\n";
}


/**
 * @details
 * Test the fft stage.
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
    std::cout << "Time taken for FFT of " << nSubbands << " subbands = "
              << double(elapsed)/double(nIter) << " ms.\n";
}


/**
 * @details
 * Test the run method.
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

    unsigned nIter = 2000;
    QTime timer;
    timer.start();
    for (unsigned i = 0; i < nIter; ++i) {
    	channeliser.run(&data, &filterCoeff, &spectra);
    }
    int elapsed = timer.elapsed();
    std::cout << "Time taken for PPF run method channelising "
    		  << " 62 subbands using 2 threads = "
              << double(elapsed)/double(nIter) << " ms.\n";
}


/**
 * @details
 * Load some ppf coefficients from file.
 */
void ChanneliserPolyphaseTest::test_loadCoeffs()
{
    unsigned nChannels = 50;
    unsigned nTaps = 10;

    QString coeffFileName = "data/coeffs_t10_c50.dat";

    PolyphaseCoefficients filterCoeffs(nTaps, nChannels);
    filterCoeffs.load(coeffFileName, nTaps, nChannels);
    CPPUNIT_ASSERT_EQUAL(nChannels, filterCoeffs.nChannels());
    CPPUNIT_ASSERT_EQUAL(nTaps, filterCoeffs.nTaps());
}




/**
 * @details
 * Contruct a spectrum and save it to file.
 */
void ChanneliserPolyphaseTest::test_makeSpectrum()
{
    // Options.
    //--------------------------------------------------------------------------
    unsigned nChannels = 64;
	unsigned nSubbands = 1;
    unsigned nPolarisations = 1;
    unsigned nSamples = nChannels;
    unsigned nTaps = 8;
    QString coeffFileName = "data/coeffs_64_1.dat";
    double freq = 10.12; // Hz
    double sampleRate = 50.0; // Hz

    TimeStreamData data(nSubbands, nPolarisations, nSamples);
    ChannelisedStreamData spectra(nSubbands, nPolarisations, nChannels);
    ConfigNode config(_configXml(nChannels));
    ChanneliserPolyphase channeliser(config);
    PolyphaseCoefficients filterCoeff(nTaps, nChannels);
    double* coeff = filterCoeff.coefficients();
    filterCoeff.load(coeffFileName, nTaps, nChannels);

    // Get object data pointers.
    //--------------------------------------------------------------------------
    std::complex<double>* timeData = data.data();

    // Generate signal.
    for (unsigned i = 0; i < nChannels; ++i) {
    	double t = double(i) / sampleRate;
    	double re = std::cos(2 * math::pi * freq * t);
    	double im = std::sin(2 * math::pi * freq * t);
    	timeData[i] = std::complex<double>(re, im);
    }

    // PPF - have to run enough times for buffer to fill with new signal.
    for (unsigned j = 0; j < 1000; ++j) {
    	channeliser.run(&data, &filterCoeff, &spectra);
    }

    // Write the spectrum.
    QFile file("spectrum.dat");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
    	return;
    }

    std::complex<double>* spectrum = spectra.data(0, 0);
    QTextStream out(&file);
    double maxFreq = sampleRate / 2.0;
    double freqInc = sampleRate / nChannels;
    for (unsigned i = 0; i < nChannels; ++i) {
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
void ChanneliserPolyphaseTest::test_channelProfile()
{
    // Options.
    //--------------------------------------------------------------------------
    unsigned nChannels = 64;
	unsigned nSubbands = 1;
    unsigned nPolarisations = 1;
    unsigned nSamples = nChannels;
    unsigned nTaps = 8;
    QString coeffFileName = "data/coeffs_64_1.dat";

    // Options controlling profile generation.
	//--------------------------------------------------------------------------
    unsigned nProfiles = 2;
	double sampleRate = 50.0; // Hz
	double startFreq = 8.0; // Hz
	unsigned nSteps = 1000;
	double freqInc = 0.01;

	// Find the channel index for the profile.
	//--------------------------------------------------------------------------
	double channelDelta = double(nChannels) / double(sampleRate);
	double endFreq = startFreq + freqInc * nSteps;
	double midTestFreq = startFreq + (endFreq - startFreq) / 2.0;
	std::cout << "scanning freqs " << startFreq << " -> " << endFreq << " ("
			  << midTestFreq << ")" << std::endl;

	//	unsigned testChannelIndex = nChannels / 2 + std::floor(midTestFreq / channelDelta);
	std::vector<unsigned> testIndices(nProfiles);
	testIndices[0] = 45;
	testIndices[1] = 46;

    // Create objects.
    //--------------------------------------------------------------------------
    try {
    	TimeStreamData data(nSubbands, nPolarisations, nSamples);
    	ChannelisedStreamData spectra(nSubbands, nPolarisations, nChannels);
    	ConfigNode config(_configXml(nChannels));
    	ChanneliserPolyphase channeliser(config);
    	PolyphaseCoefficients filterCoeff(nTaps, nChannels);
    	double* coeff = filterCoeff.coefficients();
    	filterCoeff.load(coeffFileName, nTaps, nChannels);
//    	CPPUNIT_ASSERT_DOUBLES_EQUAL(-1.0314158132996589e-04,
//    			filterCoeff.coefficients()[0], 1e-5);


    	// Get object data pointers.
    	//--------------------------------------------------------------------------
    	std::complex<double>* timeData = data.data();
    	std::complex<double>* spectrum = spectra.data(0, 0);


    	// Channel profile.
    	//--------------------------------------------------------------------------
    	std::vector<std::vector<std::complex<double> > > channelProfile;
    	channelProfile.resize(nProfiles);
    	for (unsigned i = 0; i < nProfiles; ++i) {
    		channelProfile[i].resize(nSteps);
    	}


    	// Scan frequencies to generate channel profile.
    	for (unsigned k = 0; k < nSteps; ++k) {
//    		std::cout << "loop " << k << std::endl;

    		// Generate signal.
    		double freq = startFreq + k * freqInc;
    		for (unsigned i = 0; i < nChannels; ++i) {
    			double t = double(i) / sampleRate;
    			double re = std::cos(2 * math::pi * freq * t);
    			double im = std::sin(2 * math::pi * freq * t);
    			timeData[i] = std::complex<double>(re, im);
    		}
//    		std::cout << "  - done signal gen." << std::endl;

    		// PPF - have to run enough times for buffer to fill with new signal.
    		for (unsigned j = 0; j < nTaps + 1; ++j) {
    			channeliser.run(&data, &filterCoeff, &spectra);
    		}
//    		std::cout << "  - done ppf." << std::endl;

    		// Save the amplitude of the specified channel.
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
QString ChanneliserPolyphaseTest::_configXml(unsigned nChannels,
		unsigned nThreads)
{
    QString xml =
            "<ChanneliserPolyphase>"
            "	<channels number=\"" + QString::number(nChannels) + "\"/>"
            "	<processingThreads number=\"" + QString::number(nThreads) + "\"/>"
            "</ChanneliserPolyphase>";
    return xml;
}


} // namespace lofar
} // namespace pelican
