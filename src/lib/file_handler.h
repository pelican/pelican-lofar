#ifndef FILE_HANDLER_H
#define FILE_HANDLER_H

#include "stdio.h"

typedef struct {
  char rawdatafile[80], source_name[80];
  int machine_id, telescope_id, data_type, nchans, nbits, nifs, scan_number;
  int barycentric,pulsarcentric; 
  double tstart,mjdobs,tsamp,fch1,foff,refdm,az_start,za_start,src_raj,src_dej;
  double gal_l,gal_b,header_tobs,raw_fch1,raw_foff;
  int nbeams, ibeam;

  double srcl,srcb;
  double ast0, lst0;
  long wapp_scan_number;
  char project[8];
  char culprits[24];
  double analog_power[2];

  double frequency_table[4096];
  long int npuls;

  double period;
  int nbins, itmp;
  int total_bytes;

} FILE_HEADER;

FILE_HEADER *read_header(FILE *inputfile);
unsigned long read_block(FILE *input, int nbits, float *block, unsigned long nread);

#endif // FILE_HANDLER_H
