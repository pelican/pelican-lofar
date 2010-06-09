#include "PolyphaseCoefficientsTest.h"
#include "PolyphaseCoefficients.h"

#include <QtCore/QTextStream>
#include <QtCore/QFile>
#include <iostream>
#include <iomanip>

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(PolyphaseCoefficientsTest);

PolyphaseCoefficientsTest::PolyphaseCoefficientsTest()
    : CppUnit::TestFixture()
{
}

PolyphaseCoefficientsTest::~PolyphaseCoefficientsTest()
{
}

void PolyphaseCoefficientsTest::setUp()
{
}

void PolyphaseCoefficientsTest::tearDown()
{
}


/**
 * @details
 * Tests the various accessor methods for the polyphase coefficients data blob
 */
void PolyphaseCoefficientsTest::test_accessorMethods()
{
    // Use Case
    // Construct a polyphase coefficients data blob and test each of the
	// accessor methods.
    {
        unsigned nTaps = 8;
        unsigned nChannels = 512;

        PolyphaseCoefficients coeff;
        CPPUNIT_ASSERT_EQUAL(unsigned(0), coeff.size());
        coeff.resize(nTaps, nChannels);
        CPPUNIT_ASSERT_EQUAL(nTaps * nChannels, coeff.size());
        coeff.clear();
        CPPUNIT_ASSERT_EQUAL(unsigned(0), coeff.size());
        coeff = PolyphaseCoefficients(nTaps, nChannels);
        CPPUNIT_ASSERT_EQUAL(nTaps * nChannels, coeff.size());
        CPPUNIT_ASSERT_EQUAL(nTaps, coeff.nTaps());
        CPPUNIT_ASSERT_EQUAL(nChannels, coeff.nChannels());
    }
}


/**
 * @details
 * Tests loading a coefficient file.
 */
void PolyphaseCoefficientsTest::test_loadCoeffFile()
{
    unsigned nTaps = 8;
    unsigned nChannels = 512;

    QString fileName = "thisFileDoesntExist.dat";
    PolyphaseCoefficients coeff;
    try {
    	coeff.load(fileName, nTaps, nChannels);
    }
    catch (QString err) {
		CPPUNIT_ASSERT(err.contains("Unable to open coefficients file"));
    }

    // Write a dummy test coeff file of the expected format.
    fileName = "testCoeffs.dat";
    QFile file(fileName);
	if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
		return;
	}
	QTextStream out(&file);

	for (unsigned c = 0; c < nChannels; ++c) {
		for (unsigned t = 0; t < nTaps; ++t) {
			out << QString::number(c + t, 'f', 8);
			if (t < nTaps - 1) out << " ";
		}
		out << endl;
	}
	file.close();

	// Load the file
    try {
    	coeff.load(fileName, nTaps, nChannels);
    	CPPUNIT_ASSERT_EQUAL(nTaps * nChannels, coeff.size());
    }
    catch (QString err) {
    	CPPUNIT_FAIL(err.toLatin1().data());
    }
    double* coeffs = coeff.coefficients();

    // Check some values.
    unsigned chan = 16; unsigned tap = 4;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(double(chan + tap),
    		coeffs[chan * nTaps + tap], 1.0e-5);
    chan = 423; tap = 7;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(double(chan + tap),
    		coeffs[chan * nTaps + tap], 1.0e-5);

    // Clean up.
    if (QFile::exists(fileName)) {
		QFile::remove(fileName);
    }
}


} // namespace lofar
} // namespace pelican
