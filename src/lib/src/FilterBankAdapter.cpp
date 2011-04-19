#include "FilterBankAdapter.h"
#include "SpectrumDataSet.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>


namespace pelican {

namespace lofar {


/**
 *@details FilterBankAdapter 
 */
FilterBankAdapter::FilterBankAdapter(const ConfigNode& config )
    : AbstractStreamAdapter(config)
{
    _nSamplesPerTimeBlock = config.getOption("outputChannelsPerSubband", "value", "0").toUInt();
    _nSubbands = config.getOption("subbandsPerPacket", "value", "0").toUInt();
    _nPolarisations = config.getOption("defaultPolarisations", "value", "0").toUInt();
}

/**
 *@details
 */
FilterBankAdapter::~FilterBankAdapter()
{
}

void FilterBankAdapter::deserialise(QIODevice* in)
{
        // see if its a header
        unsigned int bytes = _header.deserialise(in);

        // determine data layout
        unsigned int polarisations = _header.numberPolarisations();
        if( polarisations == 0 ) { polarisations = _nPolarisations; }
        unsigned int nSubbands = 1;
        if( nSubbands == 0 ) { nSubbands = _nPolarisations; }
        unsigned int nSamplesPerTimeBlock = _header.numberChannels();
        if( nSamplesPerTimeBlock == 0 ) { nSamplesPerTimeBlock = _nSamplesPerTimeBlock; }
        unsigned long nBlocks = (chunkSize() - bytes)/( nSubbands*polarisations*nSamplesPerTimeBlock);

        // get the object we need to fill
        SpectrumDataSetStokes* blob = (SpectrumDataSetStokes*) dataBlob();

        unsigned int nChannels = _header.numberChannels();
        blob->resize(nBlocks, _nSubbands, polarisations, _nSamplesPerTimeBlock);

        // read in block data
        for(unsigned int block=0; block < nBlocks; ++block ) {
            for(unsigned int polar=0; polar < polarisations; ++polar ) {
                for (unsigned s = 0; s < nSubbands; s++) {
                    _readBlock(in, blob->spectrumData(block, s, polar), _nSamplesPerTimeBlock);
                }
           }
        }
}

void FilterBankAdapter::_readBlock(QIODevice *in, float* block, unsigned long nread)
{
  unsigned int i; int j, k, s1, s2, s3, s4;
  unsigned short *shortblock;
  unsigned char *charblock;
  unsigned long iread;

  /* decide how to read the data based on the number of bits per sample */
  switch(_header.nbits()) {
      case 1:
          // read n/8 bytes into character block containing n 1-bit pairs
          charblock = (unsigned char *) malloc(nread / 8);
          iread = in->read((char*)charblock, nread / 8);
          k = 0;

          // unpack 1-bit pairs into the datablock
          for (i = 0; i < iread; i++)
              for (j = 0; j < 8; j++) {
                  block[k++] = charblock[i] & 1;
                  charblock[i] >>= 1;
              }
          iread = k; //number of samples read in
          free(charblock);
          break;

      case 2: // NOTE: Handles Parkes Hitrun survey data
          // read n/4 bytes into character block containing n 2-bit pairs
          charblock = (unsigned char *) malloc(nread / 4);
          iread = in->read((char*)charblock,  nread / 4 );
          j = 0;

          for(i = 0; i < iread; i++) {
              char4ints(charblock[i], &s1, &s2, &s3, &s4);
              block[j++] = (float)s1;
              block[j++] = (float)s2;
              block[j++] = (float)s3;
              block[j++] = (float)s4;
          }
          iread *= 4;
          free(charblock);
          break;

      case 4:
          // read n/2 bytes into character block containing n 4-bit pairs
          charblock = (unsigned char *) malloc(nread / 2);
          iread=in->read((char*)charblock, nread / 2 );
          j = 0;

          /* unpack 4-bit pairs into the datablock */
          for (i = 0; i < iread; i++) {
              char2ints(charblock[i], &s1, &s2);
              block[j++] = (float) s1;
              block[j++] = (float) s2;
          }
          iread *= 2; /* this is the number of samples read in */
          free(charblock);
          break;

      case 8:
          /* read n bytes into character block containing n 1-byte numbers */
          charblock = (unsigned char *) malloc(nread);
          iread = in->read((char*)charblock, nread );
          /* copy numbers into datablock */
          for (i = 0; i < iread; i++)
              block[i] = (float) charblock[i];

          free(charblock);
          break;

      case 16:
          /* read 2*n bytes into short block containing n 2-byte numbers */
          shortblock = (unsigned short *) malloc(2 * nread);
          iread = in->read((char*)shortblock, 2*nread);
          /* copy numbers into datablock */
          for (i = 0; i < iread; i++) {
              block[i] = (float) shortblock[i];
          }

          free(shortblock);
          break;

      case 32:
          /* read 4*n bytes into floating block containing n 4-byte numbers */
          iread = in->read((char*)block, 4* nread);
          break;

      default:
          std::cerr << "read_block - nbits can only be 4, 8, 16 or 32!";
  }

}

#define HI4BITS   240
#define LO4BITS   15
void FilterBankAdapter::char2ints (unsigned char c, int *i, int *j)
{
  *i =  c & LO4BITS;
  *j = (c & HI4BITS) >> 4;
}


void FilterBankAdapter::char4ints (unsigned char c, int *i, int *j, int *k, int *l)
{
  *i =  c & 3;
  *j = (c & 12) >> 2;
  *k = (c & 48) >> 4;
  *l = (c & 192) >> 6;
}


} // namespace lofar
} // namespace pelican
