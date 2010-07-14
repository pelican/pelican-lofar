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
    double* coeffs = coeff.ptr();

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


/**
 * @details Test to generate PPF coefficients.
 */
void PolyphaseCoefficientsTest::test_generate()
{
    PolyphaseCoefficients coeff;

    // Use case: Generate a window and write it to file for
    // plotting with gnuplot.
    {
        unsigned nPoints = 512 * 8;
        std::vector<double> window(nPoints);
        coeff._kaiser(nPoints, 9.0695, &window[0]);
        //coeff._gaussian(nPoints, 3.5, &window[0]);
        QFile file("window.dat");
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;
        QTextStream out(&file);
        for (unsigned i = 0; i < nPoints; ++i) {
            out << i << " " << window[i] << endl;
        }
    }

    // Use case: test generating FIR filter.
    {
        unsigned nChan = 512, nTaps = 8, nPoints = nChan * nTaps;
        std::vector<double> window(nPoints);
        coeff._kaiser(nPoints, 9.0695, &window[0]);
//        coeff._gaussian(nPoints, 9.0695, &window[0]);
//        coeff._hamming(nPoints, &window[0]);
//        coeff._blackman(nPoints, &window[0]);
        std::vector<double> filter(nPoints);
        coeff._generateFirFilter(nPoints - 1, 1.0 / nChan, &window[0], &filter[0]);
        QFile file("filter.dat");
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;
        QTextStream out(&file);
        for (unsigned i = 0; i < nPoints; ++i) {
            out << i << " " << filter[i] << endl;
        }
    }

    // Use case: test generating FIR coeffs.
    {
        unsigned nChannels = 512, nTaps = 8;
        coeff.genereateFilter(nTaps, nChannels, PolyphaseCoefficients::KAISER);
        const double* coefficients = coeff.ptr();

        QFile file("coeffs.dat");
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;
        QTextStream out(&file);
        for (unsigned c = 0; c < nChannels; ++c) {
            for (unsigned t = 0; t < nTaps; ++t) {
                out << coefficients[c * nTaps + t] << " ";
            }
            out << endl;
        }
    }
}




} // namespace lofar
} // namespace pelican
