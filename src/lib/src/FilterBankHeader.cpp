#include "FilterBankHeader.h"
#include <QtCore/QIODevice>
#include <QtCore/QByteArray>
#include <iostream>


namespace pelican {

namespace lofar {


/**
 *@details FilterBankHeader
 */
FilterBankHeader::FilterBankHeader()
{
}

/**
 *@details
 */
FilterBankHeader::~FilterBankHeader()
{
}

QString FilterBankHeader::_getString(QIODevice* device)
{
     int nchars = 0;
     device->read((char*)&nchars,sizeof(nchars));
     if( nchars <= 0 ) return QString();
     QByteArray s = device->read(nchars);
     return QString(s);
}

unsigned int FilterBankHeader::deserialise(QIODevice* device)
{
     // check the start of the stream (without destroying it)
     // to see if this is a header or not
     int c;
     device->peek((char*)&c, sizeof(c));
     if( c != 12 ) {
        // not a header
        return 0;
     }
     int bufsize = 12 + sizeof(c);
     char temp[bufsize+1];
     device->peek( &temp[0], bufsize );
     temp[bufsize]='\0';
     if (QString("HEADER_START") != QString(&temp[sizeof(int)])) {
        // not a header
        return 0;
     }

     // We have determined its a header we can now start parsing it
     device->read( bufsize );
     unsigned int totalBytes = bufsize;
     bool expecting_source_name=false;
     bool expecting_rawdatafile=false;
     bool expecting_frequency_table=false;
     int channel_index = 0;

     QString string;
     while( (string=_getString(device)).size() != 0 )  {
        totalBytes+=string.size();
        if (string=="HEADER_END") break;
        if (string == "rawdatafile" ) {
            expecting_rawdatafile=true;
        } else if (string=="source_name") {
            expecting_source_name=true;
        } else if (string=="FREQUENCY_START") {
            expecting_frequency_table=true;
            channel_index=0;
        } else if (string=="FREQUENCY_END") {
            expecting_frequency_table=false;
        } else if (string=="az_start") {
            device->read((char*)&az_start,sizeof(az_start));
            totalBytes+=sizeof(az_start);
        } else if (string=="za_start") {
            device->read((char*)&za_start,sizeof(za_start));
            totalBytes+=sizeof(za_start);
        } else if (string=="src_raj") {
            device->read((char*)&src_raj,sizeof(src_raj));
            totalBytes+=sizeof(src_raj);
        } else if (string=="src_dej") {
            device->read((char*)&src_dej,sizeof(src_dej));
            totalBytes+=sizeof(src_dej);
        } else if (string=="tstart") {
            device->read((char*)&tstart,sizeof(tstart));
            totalBytes+=sizeof(tstart);
        } else if (string=="tsamp") {
            device->read((char*)&tsamp,sizeof(tsamp));
            totalBytes+=sizeof(tsamp);
        } else if (string=="period") {
            device->read((char*)&period,sizeof(period));
            totalBytes+=sizeof(period);
        } else if (string=="fch1") {
            device->read((char*)&fch1,sizeof(fch1));
            totalBytes+=sizeof(fch1);
        } else if (string=="fchannel") {
            device->read((char*)&frequency_table[channel_index++],sizeof(double));
            totalBytes+=sizeof(double);
            fch1=foff=0.0; /* set to 0.0 to signify that a table is in use */
        } else if (string=="foff") {
            device->read((char*)&foff,sizeof(foff));
            totalBytes+=sizeof(foff);
        } else if (string=="nchans") {
            device->read((char*)&_nchans,sizeof(_nchans));
            totalBytes+=sizeof(_nchans);
        } else if (string=="telescope_id") {
            device->read((char*)&telescope_id,sizeof(telescope_id));
            totalBytes+=sizeof(telescope_id);
        } else if (string=="machine_id") {
            device->read((char*)&machine_id,sizeof(machine_id));
            totalBytes+=sizeof(machine_id);
        } else if (string=="data_type") {
            device->read((char*)&data_type,sizeof(data_type));
            totalBytes+=sizeof(data_type);
        } else if (string=="ibeam") {
            device->read((char*)&ibeam,sizeof(ibeam));
            totalBytes+=sizeof(ibeam);
        } else if (string=="nbeams") {
            device->read((char*)&nbeams,sizeof(nbeams));
            totalBytes+=sizeof(nbeams);
        } else if (string=="nbits") {
            device->read((char*)&_nbits,sizeof(_nbits));
            totalBytes+=sizeof(_nbits);
        } else if (string=="barycentric") {
            device->read((char*)&barycentric,sizeof(barycentric));
            totalBytes+=sizeof(barycentric);
        } else if (string=="pulsarcentric") {
            device->read((char*)&pulsarcentric,sizeof(pulsarcentric));
            totalBytes+=sizeof(pulsarcentric);
        } else if (string=="nbins") {
            device->read((char*)&nbins,sizeof(nbins));
            totalBytes+=sizeof(nbins);
        } else if (string=="nsamples") {
            /* read this one only for backwards compatibility */
            device->read((char*)&itmp,sizeof(itmp));
            totalBytes+=sizeof(itmp);
        } else if (string=="nifs") {
            device->read((char*)&_nifs,sizeof(_nifs));
            totalBytes+=sizeof(_nifs);
        } else if (string=="npuls") {
            device->read((char*)&npuls,sizeof(npuls));
            totalBytes+=sizeof(npuls);
        } else if (string=="refdm") {
            device->read((char*)&refdm,sizeof(refdm));
            totalBytes+=sizeof(refdm);
        } else if (expecting_rawdatafile) {
            _rawDataFile=string;
            expecting_rawdatafile=false;
        } else if (expecting_source_name) {
            _sourceName = string;
            expecting_source_name=false;
        } else {
            throw( QString("FilterBankHeader: unknown parameter: ") + string );
        }
     }
     return totalBytes;
}

} // namespace lofar
} // namespace pelican
