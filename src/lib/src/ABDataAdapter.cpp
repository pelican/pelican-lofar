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
    // set _nPol to 4?
    _nPolarisations = 4;
    blob->resize(nBlocks, 1, _nPolarisations, _nChannels);
    printf("size: %d\n", blob->size());

    // Create a temporary buffer to read out the packet headers, and
    // get the pointer to the data array in the data blob being filled.
    char headerData[_headerSize];
    char footerData[_footerSize];

    // Loop over the UDP packets in the chunk.
    char* data = NULL;
    for (unsigned p = 0; p < packets; ++p)
    {
        // Ensure there is enough data to read from the device.
        printf("a, %d\n", device->bytesAvailable());
        while (device->bytesAvailable() < _packetSize)
        {
            device->waitForReadyRead(-1);
        }
        printf("c\n");
        // Read the packet header from the input device and dump it.
        device->read(headerData, _headerSize);
        printf("d\n");

        // Build the spectrum from _packetsPerSpectrum packets.
        // Get the spectral quarter number
        unsigned int specQuart = (int) headerData[6];
        unsigned long int counter = 0;
        unsigned j = 0;
        for (signed i = 5; i >= 0; --i)
        {
            counter += ((unsigned int) headerData[i] * pow(10, j));
            ++j;
        }
        printf("e\n");
        printf("%d: %ld\n", specQuart, counter);
        unsigned bytesRead = 0;
        for (unsigned block = 0; block < nBlocks; ++block)
        {
            for (unsigned polar = 0; polar < _nPolarisations; ++polar)
            {
                data = (char*) blob->spectrumData(block, 0, polar);
                // Read the packet data from the input device into the data blob.
                bytesRead += device->read(data + bytesRead, _packetSize - _headerSize - _footerSize);

                // Read the footer and ignore it
                device->read(footerData, _footerSize);
            }
        }
        printf("done! bytesread = %d\n", bytesRead);
    }
}

