#include "file_handler.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <math.h>

#define HI4BITS   240
#define LO4BITS   15
void char2ints (unsigned char c, int *i, int *j)
{
  *i =  c & LO4BITS;
  *j = (c & HI4BITS) >> 4;
}


void char4ints (unsigned char c, int *i, int *j, int *k, int *l)
{
  *i =  c & 3;
  *j = (c & 12) >> 2;
  *k = (c & 48) >> 4;
  *l = (c & 192) >> 6;
}

int strings_equal (char *string1, char *string2) /* includefile */
{
  if (!strcmp(string1,string2)) return 1;
  else return 0;
}


// read a string from the input which looks like nchars-char[1-nchars]
void get_string(FILE *inputfile, int *nbytes, char string[]) /* includefile */
{
  int nchar;

  strcpy(string,"ERROR");
  fread(&nchar, sizeof(int), 1, inputfile);

  if (feof(inputfile)) exit(0);

  if (nchar > 80 || nchar < 1) return;

  *nbytes = sizeof(int);
  fread(string, nchar, 1, inputfile);
  string[nchar] = '\0';
  *nbytes += nchar;
}

// ================================================
// Read file header
// ================================================

/* attempt to read in the general header info from a pulsar data file */
FILE_HEADER *read_header(FILE *inputfile)
{
  char string[80], message[80];
  int nbytes, totalbytes, expecting_rawdatafile = 0, expecting_source_name = 0; 
  int channel_index;

  FILE_HEADER *header = (FILE_HEADER *) malloc(sizeof(FILE_HEADER));

  // try to read in the first line of the header
  get_string(inputfile,&nbytes,string);
  if (!strings_equal(string,"HEADER_START")) {
	// the data file is not in standard format, rewind and return
	rewind(inputfile);
	return 0;
  }
  /* store total number of bytes read so far */
  totalbytes=nbytes;

  /* loop over and read remaining header lines until HEADER_END reached */
  while (1) {
    get_string(inputfile,&nbytes,string);
   
    if (strings_equal(string,"HEADER_END")) break;
   
    totalbytes += nbytes;
    if (strings_equal(string, "rawdatafile"))
        expecting_rawdatafile = 1;
    else if (strings_equal(string, "source_name"))
        expecting_source_name = 1;
    else if (strings_equal(string, "FREQUENCY_START")) {
        channel_index = 0;
    } else if (strings_equal(string, "FREQUENCY_END"))
        ;
    else if (strings_equal(string, "az_start")) {
        fread(&(header -> az_start), sizeof(header -> az_start), 1, inputfile);
        totalbytes += sizeof(header -> az_start);
    } else if (strings_equal(string, "za_start")) {
          fread(&(header -> za_start), sizeof(header -> za_start), 1, inputfile);
          totalbytes += sizeof(header -> za_start);
    } else if (strings_equal(string, "src_raj")) {
          fread(&(header -> src_raj), sizeof(header -> src_raj), 1, inputfile);
          totalbytes += sizeof(header -> src_raj);
    } else if (strings_equal(string, "src_dej")) {
          fread(&(header -> src_dej), sizeof(header -> src_dej), 1, inputfile);
          totalbytes += sizeof(header -> src_dej);
    } else if (strings_equal(string,"tstart")) {
          fread(&(header -> tstart), sizeof(header -> tstart), 1, inputfile);
          totalbytes += sizeof(header -> tstart);
    } else if (strings_equal(string, "tsamp")) {
          fread(&(header -> tsamp), sizeof(header -> tsamp), 1, inputfile);
          totalbytes += sizeof(header -> tsamp);
    } else if (strings_equal(string, "period")) {
          fread(&(header -> period), sizeof(header -> period), 1, inputfile);
          totalbytes += sizeof(header -> period);
    } else if (strings_equal(string, "fch1")) {
          fread(&(header -> fch1), sizeof(header -> fch1), 1, inputfile);
          totalbytes += sizeof(header -> fch1);
    } else if (strings_equal(string,"fchannel")) {
          fread(&(header -> frequency_table[channel_index++]), sizeof(double), 1, inputfile);
          totalbytes += sizeof(double);
          header -> fch1 = header -> foff = 0.0; /* set to 0.0 to signify that a table is in use */
    } else if (strings_equal(string, "foff")) {
          fread(&(header -> foff), sizeof(header -> foff), 1, inputfile);
          totalbytes += sizeof(header -> foff);
    } else if (strings_equal(string, "nchans")) {
          fread(&(header -> nchans), sizeof(header -> nchans), 1, inputfile);
          totalbytes += sizeof(header -> nchans);
    } else if (strings_equal(string, "telescope_id")) {
          fread(&(header -> telescope_id), sizeof(header -> telescope_id), 1, inputfile);
          totalbytes += sizeof(header -> telescope_id);
    } else if (strings_equal(string, "machine_id")) {
          fread(&(header -> machine_id), sizeof(header -> machine_id), 1, inputfile);
          totalbytes += sizeof(header -> machine_id);
    } else if (strings_equal(string, "Telescope")) {
//          fread(&(header -> fch1), sizeof(header -> fch1), 1, inputfile);
//          totalbytes += sizeof(header -> fch1);
            get_string(inputfile,&nbytes,string);
;
    } else if (strings_equal(string, "data_type")) {
          fread(&(header -> data_type), sizeof(header -> data_type), 1, inputfile);
          totalbytes += sizeof(header -> data_type);
    } else if (strings_equal(string, "ibeam")) {
          fread(&(header -> ibeam), sizeof(header -> ibeam), 1, inputfile);
          totalbytes += sizeof(header -> ibeam);
    } else if (strings_equal(string, "nbeams")) {
          fread(&(header -> nbeams), sizeof(header -> nbeams), 1, inputfile);
          totalbytes += sizeof(header -> nbeams);
    } else if (strings_equal(string, "nbits")) {
          fread(&(header -> nbits), sizeof(header -> nbits), 1, inputfile);
          totalbytes += sizeof(header -> nbits);
    } else if (strings_equal(string, "barycentric")) {
          fread(&(header -> barycentric), sizeof(header -> barycentric), 1, inputfile);
          totalbytes += sizeof(header -> barycentric);
    } else if (strings_equal(string, "pulsarcentric")) {
          fread(&(header -> pulsarcentric), sizeof(header -> pulsarcentric), 1, inputfile);
          totalbytes += sizeof(header -> pulsarcentric);
    } else if (strings_equal(string, "nbins")) {
          fread(&(header -> nbins), sizeof(header -> nbins), 1, inputfile);
          totalbytes += sizeof(header -> nbins);
    } else if (strings_equal(string, "nsamples")) {
          /* read this one only for backwards compatibility */
          fread(&(header -> itmp), sizeof(header -> itmp), 1, inputfile);
          totalbytes += sizeof(header -> itmp);
    } else if (strings_equal(string, "nifs")) {
          fread(&(header -> nifs), sizeof(header -> nifs), 1, inputfile);
          totalbytes += sizeof(header -> nifs);
    } else if (strings_equal(string, "npuls")) {
          fread(&(header -> npuls), sizeof(header -> npuls), 1, inputfile);
          totalbytes += sizeof(header -> npuls);
    } else if (strings_equal(string, "refdm")) {
          fread(&(header -> refdm), sizeof(header -> refdm), 1, inputfile);
          totalbytes += sizeof(header -> refdm);
    } else if (expecting_rawdatafile) {
          strcpy(header -> rawdatafile, string);
          expecting_rawdatafile = 0;
    } else if (expecting_source_name) {
          strcpy(header -> source_name, string);
          expecting_source_name = 0;
    } else {
          sprintf(message, "read_header - unknown parameter: %s\n", string);
          fprintf(stderr, "ERROR: %s\n", message);
          exit(1);
    } 
  } 

  /* add on last header string */
  totalbytes += nbytes;

  /* return total number of bytes read */
  header -> total_bytes = totalbytes;
  return header;
}

// ================================================
// Read data from file
// ================================================

unsigned long read_block(FILE *input, int nbits, float *block, unsigned long nread)
{
  int i, j, k, s1, s2, s3, s4;
  unsigned short *shortblock;
  unsigned char *charblock;
  unsigned long iread;

  /* decide how to read the data based on the number of bits per sample */
  switch(nbits) {

      case 1:
          // read n/8 bytes into character block containing n 1-bit pairs
          charblock = (unsigned char *) malloc(nread / 8);
          iread = fread(charblock, 1, nread / 8, input);
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
          iread = fread(charblock, 1, nread / 4, input);
          j = 0;

          for(i = 0; i < iread; i++) {
              char4ints(charblock[i], &s1, &s2, &s3, &s4);
              block[j++] = (float) s1;
              block[j++] = (float) s2;
              block[j++] = (float) s3;
              block[j++] = (float) s4;
          }
 
          iread *= 4;        
          free(charblock);
          break;

      case 4:
          // read n/2 bytes into character block containing n 4-bit pairs
          charblock = (unsigned char *) malloc(nread / 2);
          iread=fread(charblock, 1, nread / 2, input);
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
          iread = fread(charblock, 1, nread, input);
          /* copy numbers into datablock */
          for (i = 0; i < iread; i++)
              block[i] = (float) charblock[i];

          free(charblock);
          break;

      case 16:
          /* read 2*n bytes into short block containing n 2-byte numbers */
          shortblock = (unsigned short *) malloc(2 * nread);
          iread = fread(shortblock, 2, nread, input);
      
          /* copy numbers into datablock */
          for (i = 0; i < iread; i++) {
              block[i] = (float) shortblock[i];
          }

          free(shortblock);
          break;

      case 32:
          /* read 4*n bytes into floating block containing n 4-byte numbers */
          iread = fread(block, 4, nread, input); 
          break;

      default:
          fprintf(stderr, "read_block - nbits can only be 4, 8, 16 or 32!");
  }

  /* return number of samples read */
  return iread;
}
