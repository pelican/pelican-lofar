#ifndef FILTERBANKADAPTER_H
#define FILTERBANKADAPTER_H


#include "pelican/core/AbstractStreamAdapter.h"
#include "FilterBankHeader.h"

/**
 * @file FilterBankAdapter.h
 */

namespace pelican {

namespace lofar {
class SpectrumDataSetStokes;

/**
 * @class FilterBankAdapter
 *  
 * @brief
 *    Adapt a SigProc Filterbank Format stream
 *    into a StreamDataStokes data format
 * @details
 * 
 */

class FilterBankAdapter : public AbstractStreamAdapter
{
    public:
        FilterBankAdapter( const ConfigNode& config  );
        ~FilterBankAdapter();

        /// Method to deserialise a LOFAR time stream data.
        void deserialise(QIODevice* in);

    private:
        void _readBlock(QIODevice *input, float* , unsigned long nread);
        void char4ints (unsigned char c, int *i, int *j, int *k, int *l);
        void char2ints (unsigned char c, int *i, int *j);

    private:
        FilterBankHeader _header;
        unsigned int _nSamplesPerTimeBlock;
        unsigned int _nPolarisations;
        unsigned int _nSubbands;
};

PELICAN_DECLARE_ADAPTER(FilterBankAdapter)

} // namespace lofar
} // namespace pelican
#endif // FILTERBANKADAPTER_H 
