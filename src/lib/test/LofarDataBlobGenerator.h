#ifndef LOFARDATABLOBGENERATOR_H
#define LOFARDATABLOBGENERATOR_H


#include "pelican/core/AbstractDataClient.h"
#include "pelican/data/DataSpec.h"
#include <QList>

/**
 * @file LofarDataBlobGenerator.h
 */

namespace pelican {

namespace lofar {
class TimeSeriesDataSetC32;

/**
 * @class LofarDataBlobGenerator
 *  
 * @brief
 *     Provides a DataClient that generates Lofar specific DataBlobs
 *     for testing purposes
 * @details
 * 
 */

class LofarDataBlobGenerator : public AbstractDataClient
{
    public:
        LofarDataBlobGenerator( const ConfigNode& configNode,
                                const DataTypes& types, const Config* config );
        ~LofarDataBlobGenerator();

        /// required interface for DataClient
        virtual DataBlobHash getData(DataBlobHash&);
        virtual const DataSpec& dataSpec() const;

        /// set the number of channels
        void setChannels( unsigned nChannels ) { _nChannels = nChannels; }

        /// set the number of subbands
        void setSubbands( unsigned num ) { _nSubbands= num; }

        /// set the number of ipolarisations
        void setPolarisations( unsigned num ) { _nPols = num; }

        /// set the number of ipolarisations
        void setTimeSamplesPerBlock( unsigned num ) { _nSamples = num; }


    private:
        TimeSeriesDataSetC32* generateTimeSeriesData( 
                                TimeSeriesDataSetC32* timeSeries ) const;

    private:
        unsigned _nChannels;
        unsigned _nSubbands;
        unsigned _nPols;
        unsigned _nSamples;
        unsigned _counter;
        mutable QList<TimeSeriesDataSetC32*> _readyMade;
        DataSpec _availableStreams;
};
PELICAN_DECLARE_CLIENT(LofarDataBlobGenerator)

} // namespace lofar
} // namespace pelican
#endif // LOFARDATABLOBGENERATOR_H 
