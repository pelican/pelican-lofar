#include "SpectrumDataSetTest.h"
#include "SpectrumDataSet.h"

#include "pelican/utility/FactoryGeneric.h"
#include "timer.h"

#include <QtCore/QBuffer>

#include <iostream>
#include <complex>

using std::cout;
using std::cerr;
using std::endl;
using std::vector;
typedef std::complex<float> Complex;

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(SpectrumDataSetTest);

/**
 * @details
 * Tests the various accessor methods for the time stream data blob
 */
void SpectrumDataSetTest::test_accessorMethods()
{
    unsigned nTimeBlocks = 10;
    unsigned nSubbands = 62;
    unsigned nPols = 2;
    unsigned nChan = 10;

    // Use Case
    // Construct a sub-band spectra data blob directly.
    {
        try {
            SpectrumDataSet<float> spectra;

            // Resize the data blob.
            spectra.resize(nTimeBlocks, nSubbands, nPols, nChan);

            // Check the dimension accessor methods.
            CPPUNIT_ASSERT_EQUAL(nTimeBlocks, spectra.nTimeBlocks());
            CPPUNIT_ASSERT_EQUAL(nSubbands, spectra.nSubbands());
            CPPUNIT_ASSERT_EQUAL(nPols, spectra.nPolarisations());
            CPPUNIT_ASSERT_EQUAL(nChan, spectra.nChannels());

            // Expect to throw as base class methods have not been implemented.
            CPPUNIT_ASSERT_THROW(spectra.serialisedBytes(), QString);
            QBuffer buffer;
            CPPUNIT_ASSERT_THROW(spectra.serialise(buffer), QString);
            CPPUNIT_ASSERT_THROW(spectra.deserialise(buffer, QSysInfo::ByteOrder),
                    QString);
        }
        catch (const QString& err)
        {
            cout << err.toStdString() << endl;
        }
    }

    // Use Case
    // Construct a subband spectra data blob using the factory
    {
        try {
            FactoryGeneric<DataBlob> factory;
            DataBlob* s = factory.create("SpectrumDataSetC32");

            SpectrumDataSetC32* spectra = (SpectrumDataSetC32*)s;
            spectra->resize(10, 62, 2, 10);

            CPPUNIT_ASSERT_EQUAL(nTimeBlocks, spectra->nTimeBlocks());
            CPPUNIT_ASSERT_EQUAL(nSubbands, spectra->nSubbands());
            CPPUNIT_ASSERT_EQUAL(nPols, spectra->nPolarisations());
            CPPUNIT_ASSERT_EQUAL(nChan, spectra->nChannels());
        }
        catch (const QString& err)
        {
            cout << err.toStdString() << std::endl;
        }
    }

}


/**
 *
 */
void SpectrumDataSetTest::test_serialise_deserialise()
{
     // Error tolerance use for double comparisons.
     double err = 1.0e-5;

     // Create a spectra blob.
     unsigned nTimeBlocks = 10;
     unsigned nSubbands = 5;
     unsigned nPolarisations = 2;
     unsigned nChannels = 10;
     SpectrumDataSetC32 spectra;

     spectra.resize(nTimeBlocks, nSubbands, nPolarisations, nChannels);

     CPPUNIT_ASSERT_EQUAL(nTimeBlocks * nSubbands * nPolarisations,
             spectra.nSpectra());

     Complex* data;
     for (unsigned i = 0; i < spectra.nSpectra(); ++i)
     {
         data = spectra.spectrumData(i);
         CPPUNIT_ASSERT(data);
         for (unsigned c = 0; c < nChannels; ++c)
         {
             data[c] = Complex(float(i) + float(c), float(i) - float(c));
         }
     }

     long blockRate = 1010;
     long long timeStamp = 919191;
     spectra.setBlockRate(blockRate);
     spectra.setLofarTimestamp(timeStamp);

     // Serialise to a QBuffer.
     QBuffer serialBlob;
     serialBlob.open(QBuffer::WriteOnly);
     spectra.serialise(serialBlob);

     // Check the size of the byte array is the same as reported.
     CPPUNIT_ASSERT_EQUAL((qint64)spectra.serialisedBytes(), serialBlob.size());
     serialBlob.close();

     serialBlob.open(QBuffer::ReadOnly);

     // Deserialise into a new spectra data blob.
     SpectrumDataSetC32 spectraNew;
     spectraNew.deserialise(serialBlob, QSysInfo::ByteOrder);

     // Check we read everything.
     CPPUNIT_ASSERT(serialBlob.bytesAvailable() == 0);
     serialBlob.close();

     // Check the blob deserialised correctly.
     CPPUNIT_ASSERT_EQUAL(nTimeBlocks, spectraNew.nTimeBlocks());
     CPPUNIT_ASSERT_EQUAL(nSubbands, spectraNew.nSubbands());
     CPPUNIT_ASSERT_EQUAL(nPolarisations, spectraNew.nPolarisations());
     CPPUNIT_ASSERT_EQUAL(nChannels, spectraNew.nChannels());

     const Complex* dataNew;
     for (unsigned i = 0; i < spectraNew.nSpectra(); ++i)
     {
         dataNew = spectraNew.spectrumData(i);

         for (unsigned c = 0; c < nChannels; ++c)
         {
             CPPUNIT_ASSERT_DOUBLES_EQUAL(float(i) + float(c), dataNew[c].real(), err);
             CPPUNIT_ASSERT_DOUBLES_EQUAL(float(i) - float(c), dataNew[c].imag(), err);
         }
     }

     CPPUNIT_ASSERT_EQUAL(blockRate, spectraNew.getBlockRate());
     CPPUNIT_ASSERT_EQUAL(timeStamp, spectraNew.getLofarTimestamp());
}


void SpectrumDataSetTest::test_access_performance()
{
//    unsigned nTimeBlocks = 16384;
//    unsigned nSubbands = 62;
//    unsigned nPolarisations = 2;
//    unsigned nChannels = 16;
//
//    SpectrumDataSetC32 * spectra = new SpectrumDataSetC32;
//    spectra->resize(nTimeBlocks, nSubbands, nPolarisations, nChannels);

}


} // namespace lofar
} // namespace pelican
