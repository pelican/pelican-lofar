#include "ChannelisedStreamDataTest.h"
#include "ChannelisedStreamData.h"

#include "pelican/utility/FactoryGeneric.h"

#include <iostream>
#include <complex>
#include <QtCore/QBuffer>

namespace pelican {
namespace lofar {

CPPUNIT_TEST_SUITE_REGISTRATION( ChannelisedStreamDataTest );

ChannelisedStreamDataTest::ChannelisedStreamDataTest()
    : CppUnit::TestFixture()
{
}

ChannelisedStreamDataTest::~ChannelisedStreamDataTest()
{
}

void ChannelisedStreamDataTest::setUp()
{
}

void ChannelisedStreamDataTest::tearDown()
{
}


/**
 * @details
 * Tests the various accessor methods for the time stream data blob
 */
void ChannelisedStreamDataTest::test_accessorMethods()
{
    // Use Case
    // Construct an channelised stream data blob and test each of the accessor methods.
    {
        // Error tolerance use for double comparisons.
        double err = 1.0e-5;

        // Check default constructor
        ChannelisedStreamData spectrum;

        // Check resize
        unsigned nSubbands = 3;
        unsigned nPolarisations = 2;
        unsigned nChannels = 10;
        spectrum.resize(nSubbands, nPolarisations, nChannels);
        CPPUNIT_ASSERT_EQUAL(nSubbands * nPolarisations * nChannels, spectrum.size());

        // Check clear
        spectrum.clear();
        CPPUNIT_ASSERT_EQUAL(unsigned(0), spectrum.size());

        // Check the constructor that takes a size.
        nChannels = 6;
        spectrum = ChannelisedStreamData(nSubbands, nPolarisations, nChannels);
        CPPUNIT_ASSERT_EQUAL(nSubbands * nPolarisations * nChannels, spectrum.size());

        // Check dimension return methods.
        CPPUNIT_ASSERT_EQUAL(nSubbands, spectrum.nSubbands());
        CPPUNIT_ASSERT_EQUAL(nPolarisations, spectrum.nPolarisations());
        CPPUNIT_ASSERT_EQUAL(nChannels, spectrum.nChannels());

        // Check the start frequency.
        double startFreq = 0.1;
        spectrum.setStartFrequency(startFreq);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(startFreq, spectrum.startFrequency(), err);

        // Check the sample delta.
        double channelFrequencyDelta = 0.22;
        spectrum.setChannelfrequencyDelta(channelFrequencyDelta);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(channelFrequencyDelta,
                spectrum.channelFrequencyDelta(), err);

        // Check that the data pointer is not null when the data blob is sized.
        CPPUNIT_ASSERT(spectrum.data() != NULL);

        // Check clearing of metadata.
        spectrum.clear();
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, spectrum.channelFrequencyDelta(), err);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, spectrum.startFrequency(), err);
        CPPUNIT_ASSERT_EQUAL(unsigned(0), spectrum.nSubbands());
        CPPUNIT_ASSERT_EQUAL(unsigned(0), spectrum.nPolarisations());
        CPPUNIT_ASSERT_EQUAL(unsigned(0), spectrum.nChannels());

        // Check that the data pointer is null when the data blob is empty.
        CPPUNIT_ASSERT(spectrum.data() == NULL);

        // Resize and add some data.
        spectrum.resize(nSubbands, nPolarisations, nChannels);
        std::complex<double>* in = spectrum.data();
        for (unsigned i = 0, sb = 0; sb < nSubbands; ++sb) {
            for (unsigned p = 0; p < nPolarisations; ++p) {
                for (unsigned c = 0; c < nChannels; ++c) {
                    double re = double(sb) + double(p) + double(c);
                    double im = double(c) - double(p) - double(sb);
                    in[i] = std::complex<double>(re, im);
                    i++;
                }
            }
        }

        const std::complex<double>* out = spectrum.data();
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, out[0].real(), err);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, out[0].imag(), err);
        unsigned sb = 1; // sub-band
        unsigned p = 0;  // polarisation
        unsigned c = 2;  // channel
        unsigned index =  sb * nPolarisations * nChannels + p * nChannels + c;
        CPPUNIT_ASSERT_EQUAL(index, spectrum.index(sb, p, c));
        CPPUNIT_ASSERT(index < spectrum.size());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(sb) + double(p) + double(c), out[index].real(), err);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(c) - double(p) - double(sb), out[index].imag(), err);

        sb = 1; p = 0; c = 0;
        // Check pointer to sub-band
        const std::complex<double>* sb1 = spectrum.data(sb);
        CPPUNIT_ASSERT(sb1 == &out[spectrum.index(sb, p, c)]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(sb) + double(p) + double(c), sb1[0].real(), err);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(c) - double(p) - double(sb), sb1[0].imag(), err);

        sb = 2; p = 1; c = 0;
        const std::complex<double>* sb2p1 = spectrum.data(sb, p);
        CPPUNIT_ASSERT(sb2p1 == &out[spectrum.index(sb, p, c)]);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(sb) + double(p) + double(c), sb2p1[0].real(), err);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(c) - double(p) - double(sb), sb2p1[0].imag(), err);

        c = 3;
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(sb) + double(p) + double(c), sb2p1[c].real(), err);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(c) - double(p) - double(sb), sb2p1[c].imag(), err);

//        spectrum.write("spectrum.txt");

    }

    {
        FactoryGeneric<DataBlob> factory;
        DataBlob* d = factory.create("ChannelisedStreamData");
        ChannelisedStreamData* c = (ChannelisedStreamData*)d;
        c->resize(10, 2, 512);
    }
}


/**
 * @details
 * Tests of the serialise and deserialise methods.
 */
void ChannelisedStreamDataTest::test_serialise_deserialise()
{
    // Error tolerance use for double comparisons.
     double err = 1.0e-5;

     // Construct a blob and fill in some data.
     unsigned nSubbands = 3;
     unsigned nPolarisations = 2;
     unsigned nChannels = 10;
     ChannelisedStreamData spectrum1(nSubbands, nPolarisations, nChannels);
     std::complex<double>* in = spectrum1.data();
     for (unsigned i = 0, sb = 0; sb < nSubbands; ++sb) {
         for (unsigned p = 0; p < nPolarisations; ++p) {
             for (unsigned c = 0; c < nChannels; ++c) {
                 double re = double(i);
                 double im = double(sb + p + c);
                 in[i] = std::complex<double>(re, im);
                 i++;
             }
         }
     }
     double startFreq = 1.01020304e6;
     double freqDelta = 3.456789e2;
     spectrum1.setStartFrequency(startFreq);
     spectrum1.setChannelfrequencyDelta(freqDelta);

     // Serialise to a QBuffer (QIODevice).
     QBuffer serialBlob;
     serialBlob.open(QBuffer::WriteOnly);
     spectrum1.serialise(serialBlob);
     serialBlob.close();

     // check the return byte array is the expected size.
     qint64 expectedSize = 3 * sizeof(unsigned) + 2 * sizeof(double)
             + spectrum1.size() * sizeof(std::complex<double>);
     CPPUNIT_ASSERT_EQUAL(expectedSize, serialBlob.size());

     serialBlob.open(QBuffer::ReadOnly);

     // Construct a new data blob to fill via the deserialise.
     ChannelisedStreamData spectrum2(serialBlob);

     CPPUNIT_ASSERT(serialBlob.bytesAvailable() == 0);
     serialBlob.close();

     // Check the header deserialised correctly.
     CPPUNIT_ASSERT_EQUAL(nSubbands, spectrum2.nSubbands());
     CPPUNIT_ASSERT_EQUAL(nPolarisations, spectrum2.nPolarisations());
     CPPUNIT_ASSERT_EQUAL(nChannels, spectrum2.nChannels());
     CPPUNIT_ASSERT_DOUBLES_EQUAL(startFreq, spectrum2.startFrequency(), err);
     CPPUNIT_ASSERT_DOUBLES_EQUAL(freqDelta, spectrum2.channelFrequencyDelta(), err);

     const std::complex<double>* out = spectrum2.data();
     for (unsigned i = 0, sb = 0; sb < nSubbands; ++sb) {
         for (unsigned p = 0; p < nPolarisations; ++p) {
             for (unsigned c = 0; c < nChannels; ++c) {
                 double re = double(i);
                 double im = double(sb + p + c);
                 CPPUNIT_ASSERT_DOUBLES_EQUAL(re, out[i].real(), err);
                 CPPUNIT_ASSERT_DOUBLES_EQUAL(im, out[i].imag(), err);
                 i++;
             }
         }
     }
}


} // namespace lofar
} // namespace pelican
