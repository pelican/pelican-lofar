#include "LofarDataBlobGenerator.h"
#include "TimeSeriesDataSet.h"
#include <QDebug>


namespace pelican {

namespace lofar {


/**
 *@details LofarDataBlobGenerator 
 */
LofarDataBlobGenerator::LofarDataBlobGenerator( const ConfigNode& configNode,
                                                const DataTypes& types, const Config* config
                                              )
    : AbstractDataClient( configNode, types, config )
{
}

/**
 *@details
 */
LofarDataBlobGenerator::~LofarDataBlobGenerator()
{
}

AbstractDataClient::DataBlobHash LofarDataBlobGenerator::getData(
        AbstractDataClient::DataBlobHash& dataHash ) {
    DataBlobHash validHash;

    qDebug() << "getData:" << dataHash.keys();
    foreach(const DataRequirements& req, dataRequirements()) {
        foreach(const QString& type, req.serviceData())
        {
            if( ! dataHash.contains(type) )
                throw( QString("LofarDataBlobGenerator: getData() called without DataBlob %1").arg(type) );
        }
        foreach(const QString& type, req.streamData() )
        {
            if( ! dataHash.contains(type) )
                throw( QString("FileDataClient: getData() called without DataBlob %1").arg(type) );
            validHash.insert( type, generateTimeSeriesData( 
                dynamic_cast<TimeSeriesDataSetC32*>(dataHash.value(type)) ) );
        }
    }
    return validHash;
}


TimeSeriesDataSetC32* LofarDataBlobGenerator::generateTimeSeriesData( TimeSeriesDataSetC32* timeSeries ) const {
    unsigned timesPerChunk = 16 * 16384;
    if (timesPerChunk % _nChannels)
        Q_ASSERT("Setup error");
    unsigned nBlocks = timesPerChunk / _nChannels;
    timeSeries->resize( nBlocks, _nSubbands, _nPols, _nChannels );
}

} // namespace lofar
} // namespace pelican
