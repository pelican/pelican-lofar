#include "SpectrumDataSetTest.h"
#include "SpectrumDataSet.h"

#include "pelican/utility/FactoryGeneric.h"

#include <QtCore/QBuffer>

#include <iostream>
#include <complex>

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION(SpectrumDataSetTest);

/**
 * @details
 * Tests the various accessor methods for the time stream data blob
 */
void SpectrumDataSetTest::test_accessorMethods()
{
    // Use Case
    // Construct a sub-band spectra data blob directly
    {
        try {
            SpectrumDataSet<float> spectra;

            spectra.resize(10, 62, 2, 1);
            CPPUNIT_ASSERT_EQUAL(unsigned(10), spectra.nTimeBlocks());
            CPPUNIT_ASSERT_EQUAL(unsigned(62), spectra.nSubbands());
            CPPUNIT_ASSERT_EQUAL(unsigned(2), spectra.nPolarisations());

            // Expect to throw as base class methods have not been implemented.
            CPPUNIT_ASSERT_THROW(spectra.serialisedBytes(), QString);
            QBuffer buffer;
            CPPUNIT_ASSERT_THROW(spectra.serialise(buffer), QString);
            CPPUNIT_ASSERT_THROW(spectra.deserialise(buffer, QSysInfo::ByteOrder),
                    QString);
        }
        catch (const QString& err) {
            std::cout << err.toStdString() << std::endl;
        }
    }

    // Use Case
    // Construct a subband spectra data blob using the factory
    {
        try {
            FactoryGeneric<DataBlob> factory;
            DataBlob* s = factory.create("SpectrumDataSetC32");

            SpectrumDataSetC32* spectra = (SpectrumDataSetC32*)s;
            spectra->resize(10, 62, 2, 1);

            CPPUNIT_ASSERT_EQUAL(unsigned(10), spectra->nTimeBlocks());
            CPPUNIT_ASSERT_EQUAL(unsigned(62), spectra->nSubbands());
            CPPUNIT_ASSERT_EQUAL(unsigned(2), spectra->nPolarisations());
        }
        catch (const QString& err) {
            std::cout << err.toStdString() << std::endl;
        }
    }

}


/**
 *
 */
void SpectrumDataSetTest::test_serialise_deserialise()
{
//     // Error tolerance use for double comparisons.
//     double err = 1.0e-5;

//     // Create a spectra blob.
//     unsigned nTimeBlocks = 10;
//     unsigned nSubbands = 5;
//     unsigned nPolarisations = 2;
//     SpectrumDataSetC32 spectra;
//     spectra.resize(nTimeBlocks, nSubbands, nPolarisations, 1);
//     CPPUNIT_ASSERT_EQUAL(nTimeBlocks * nSubbands * nPolarisations,
//             spectra.nSpectra());

//     for (unsigned i = 0; i < spectra.nSpectra(); ++i) {
//         Spectrum<std::complex<float> >* spectrum = spectra.spectrum(i);
//         CPPUNIT_ASSERT(spectrum != NULL);

//         spectrum->setStartFrequency(double(i) + 0.1);
//         spectrum->setFrequencyIncrement(double(i) + 0.2);
//         unsigned nChannels = 10;
//         spectrum->resize(nChannels);
//         std::complex<float>* channelAmp = spectrum->data();
//         for (unsigned c = 0; c < spectrum->nChannels(); ++c) {
//             channelAmp[c] = std::complex<float>(float(i) + float(c),
//                     float(i) - float(c));
//         }
//     }

//     // Serialise to a QBuffer.
//     QBuffer serialBlob;
//     serialBlob.open(QBuffer::WriteOnly);
//     spectra.serialise(serialBlob);

//     // Check the size of the byte array is the same as reported.
//     CPPUNIT_ASSERT_EQUAL((qint64)spectra.serialisedBytes(), serialBlob.size());
//     serialBlob.close();

//     serialBlob.open(QBuffer::ReadOnly);

//     // Deserialise into a new spectra data blob.
//     SpectrumDataSetC32 spectraNew;
//     spectraNew.deserialise(serialBlob, QSysInfo::ByteOrder);

//     // Check we read everything.
//     CPPUNIT_ASSERT(serialBlob.bytesAvailable() == 0);
//     serialBlob.close();

//     // Check the blob deserialised correctly.
//     CPPUNIT_ASSERT_EQUAL(nTimeBlocks, spectraNew.nTimeBlocks());
//     CPPUNIT_ASSERT_EQUAL(nSubbands, spectraNew.nSubbands());
//     CPPUNIT_ASSERT_EQUAL(nPolarisations, spectraNew.nPolarisations());

//     for (unsigned i = 0; i < spectraNew.nSpectra(); ++i) {
//         const Spectrum<std::complex<float> >* spectrum = spectra.spectrum(i);
//         unsigned nChannels = spectrum->nChannels();
//         CPPUNIT_ASSERT_EQUAL(10u, nChannels);
//         const std::complex<float>* channelAmp = spectrum->data();
//         for (unsigned c = 0; c < nChannels; ++c) {
//             CPPUNIT_ASSERT_DOUBLES_EQUAL(float(i) + float(c),
//                     channelAmp[c].real(), err);
//             CPPUNIT_ASSERT_DOUBLES_EQUAL(float(i) - float(c),
//                     channelAmp[c].imag(), err);
//         }
//     }
}

} // namespace lofar
} // namespace pelican
