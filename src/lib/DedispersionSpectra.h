#ifndef DEDISPERSIONSPECTRA_H
#define DEDISPERSIONSPECTRA_H


#include "pelican/data/DataBlob.h"
#include "BinMap.h"
#include <QVector>

/**
 * @file DedispersionSpectra.h
 */

namespace pelican {

namespace lofar {

/**
 * @class DedispersionSpectra
 *  
 * @brief
 *     Stokes x DM x time
 * @details
 *    The dm spectra is a function of DM value vs. the total integral of the power output at that dm
 *    This class reprepesnts a collection of these spectra, one for each time slice
 */

class DedispersionSpectra : public DataBlob
{
    public:
        DedispersionSpectra();
        void resize( unsigned timebins, unsigned dedispersionBins, 
                     float dedispersionBinStart, float dedispersionBinWidth );
        ~DedispersionSpectra();

        /// return the Dedispersion (dm vs. integrated power from freq-time data)
        //const DedispersionData<T>& spectra( unsigned timeBin ) const;

        /// return the entire data chunk as a contiguous block
        //  represented as a series of vectors for each time stamp. The vector 
        //  represents the summed power for
        //  each dedispersion value.
        inline QVector<float>& data() { return _data; }

        /// the inegrated power for the specified dm and timeslice
        float dm( unsigned timeSlice, float dm ) const;

        /// return the index of the bin for a given dm value
        unsigned int dmIndex( float dm ) const;

    private:
        BinMap _dmBin;
        unsigned _timeBins;
        unsigned _dedispersionBins;
        QVector<float> _data;
};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONSPECTRA_H 
