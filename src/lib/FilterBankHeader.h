#ifndef FILTERBANKHEADER_H
#define FILTERBANKHEADER_H

class QIODevice;
#include <QString>

/**
 * @file FilterBankHeader.h
 */

namespace pelican {

namespace lofar {

/**
 * @class FilterBankHeader
 *  
 * @brief
 *    FilterBank Format Header information
 * @details
 * 
 */

class FilterBankHeader
{
    public:
        FilterBankHeader(  );
        ~FilterBankHeader();
        /// deserialise from an incoming IO device
        // returns the number of bytes read from the stream
        unsigned int deserialise(QIODevice *stream);

        inline int nbits() const { return _nbits; };
        inline unsigned int numberPolarisations() { return _nifs; }
        inline unsigned int numberChannels() { return _nchans; }

    protected:
        QString _getString(QIODevice*);

    private:
        QString _sourceName;
        QString _rawDataFile;
        //char rawdatafile[80], source_name[80];
        int machine_id, telescope_id, data_type, _nbits, scan_number;
        unsigned int _nifs, _nchans;
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

};

} // namespace lofar
} // namespace pelican
#endif // FILTERBANKHEADER_H 
