#include "ABDataAdapter.h"
#include "SpectrumDataSet.h"
//jtest
#include <stdio.h>

using namespace pelican;
using namespace pelican::ampp;

// Construct the signal data adapter.
ABDataAdapter::ABDataAdapter(const ConfigNode& config)
    : AbstractStreamAdapter(config)
{
    // Read the configuration using configuration node utility methods.
    _packetsPerSpectrum = config.getOption("spectrum", "packets").toUInt();
    _channelsPerPacket = config.getOption("packet", "channels").toUInt();

    // Set up the packet data.
    _packetSize = _headerSize + _channelsPerPacket * 8 + _footerSize;

   // Calculate the total number of channels.
   _nChannels = _packetsPerSpectrum * _channelsPerPacket;
}

// Called to de-serialise a chunk of data from the input device.
void ABDataAdapter::deserialise(QIODevice* device)
{
    // A pointer to the data blob to fill should be obtained by calling the
    // dataBlob() inherited method. This returns a pointer to an
    // abstract DataBlob, which should be cast to the appropriate type.
    SpectrumDataSetStokes* blob = (SpectrumDataSetStokes*) dataBlob();

    // Set the size of the data blob to fill.
    // The chunk size is obtained by calling the chunkSize() inherited method.
    unsigned packets = chunkSize() / _packetSize;
    // Number of time samples; Each channel contains 4 psuedo-Stokes values,
    // each of size sizeof(short int)
    unsigned nBlocks = packets / _packetsPerSpectrum;
    _nPolarisations = 4;
    blob->resize(nBlocks, 1, _nPolarisations, _nChannels);
    //printf("#deserialize, %d, %d, %d, %d\n", nBlocks, _nPolarisations, _nChannels, blob->size());

    // Create a temporary buffer to read out the packet headers, and
    // get the pointer to the data array in the data blob being filled.
    char headerData[_headerSize];
    char d[_packetsPerSpectrum * (_packetSize - _headerSize - _footerSize)];
    char footerData[_footerSize];

    // Loop over the UDP packets in the chunk.
    float* data = NULL;
    unsigned bytesRead = 0;
    unsigned block = 0;
    signed int specQuart = 0;
    signed int prevSpecQuart = -1;
    unsigned long int counter = 0;
    for (unsigned p = 0; p < packets; ++p)
    {
        // Ensure there is enough data to read from the device.
        while (device->bytesAvailable() < _packetSize)
        {
            device->waitForReadyRead(-1);
        }
        // Read the packet header from the input device and dump it.
        device->read(headerData, _headerSize);

        // Build the spectrum from _packetsPerSpectrum packets and write the
        // data out to the blob.
        // Get the spectral quarter number
        specQuart = (signed char) headerData[6];
        if (specQuart - prevSpecQuart != 1)
        {
            fprintf(stderr,
                    "# Missing packet! specQuart = %d, prevSpecQuart = %d\n",
                    specQuart,
                    prevSpecQuart);
        }
        if (_packetsPerSpectrum - 1 == specQuart)
        {
            prevSpecQuart = -1;
        }
        else
        {
            prevSpecQuart = specQuart;
        }
        counter = (*(unsigned long int *) headerData)
                                    & 0x0000FFFFFFFFFFFF;
        //printf("%lu\n", counter);
        bytesRead += device->read(d + bytesRead,
                                  _packetSize - _headerSize - _footerSize);
        //printf("%d\n", bytesRead);
        // Write out spectrum to blob if this is the last spectral quarter.
        short int* dd = (short int*) d;
        if (_packetsPerSpectrum - 1 == specQuart)
        {
            #if 0
            for (unsigned i = 0; i < (_packetSize - _headerSize - _footerSize) * 4 / sizeof(short int); i += 4)
            {
                fprintf(stderr, "%ld, %d\n", &dd[i], dd[i]);
            }
            #endif
            for (unsigned polar = 0; polar < _nPolarisations; ++polar)
            {
                data = (float*) blob->spectrumData(block, 0, polar);
                for (unsigned chan = 0; chan < _nChannels; ++chan)
                {
                    data[chan] = (float) dd[chan * 4 + polar];
                    #if 0
                    if (0 == polar)
                    {
                        //printf("%ld, %d\n", &dd[chan*4+polar], dd[chan*4+polar]);
                        printf("%g\n", data[chan]);
                    }
                    #endif
                }
            }
            memset(d, '\0', _packetsPerSpectrum * (_packetSize - _headerSize - _footerSize));
            bytesRead = 0;
            ++block;
            Q_ASSERT(block <= nBlocks);
        }
        // Read the packet footer from the input device and dump it.
        device->read(footerData, _footerSize);
    }
}

