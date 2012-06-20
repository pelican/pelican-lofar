#include "LofarDataBlobGenerator.h"
#include "TimeSeriesDataSet.h"
#include "constants.h"
#include <QDebug>
#include <vector>
#include <cmath>


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
    _nPols= 2;
    _nSubbands = 64;
    _nChannels = 32;
    _nSamples = 6400;
    _counter = 0;
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

    foreach(const DataRequirements& req, dataRequirements()) {
        foreach(const QString& type, req.serviceData() )
        {
            if( ! dataHash.contains(type) )
                throw( QString("LofarDataBlobGenerator: getData() called without DataBlob %1").arg(type) );
        }
        foreach(const QString& type, req.streamData() )
        {
            if( ! dataHash.contains(type) )
                throw( QString("LofarDataBlobGenerator: getData() called without DataBlob %1").arg(type) );
            ++_counter;
            validHash.insert( type, generateTimeSeriesData( 
                dynamic_cast<TimeSeriesDataSetC32*>(dataHash.value(type)) ) );
        }
    }
    return validHash;
}


TimeSeriesDataSetC32* LofarDataBlobGenerator::generateTimeSeriesData( TimeSeriesDataSetC32* timeSeries ) const {
    // if weve seen this object before don't regenrate it 
    // as filling DataBlob is very expensive
    if ( _readyMade.contains( timeSeries ) ) return timeSeries;
    qDebug() << "Generating Data for TimeSeriesDataSet" << timeSeries ;
    if (_nSamples % _nChannels)
        throw QString("Setup error: nSamples must be multiple of nChannles");
    unsigned nBlocks = _nSamples / _nChannels;
    timeSeries->resize( nBlocks, _nSubbands, _nPols, _nChannels );

    // Generate channel profile by scanning though frequencies.
    unsigned nSteps   = 1000;     // Number of steps in profile.
    double sampleRate = 50.0e6; // Hz
    double startFreq  = 8.0e6;   // Hz
    double freqInc    = 0.01e6;    // Frequency increment of profile steps.
    std::vector<double> freqs(nSteps);

    for (unsigned k = 0; k < nSteps; ++k)
    {
        // Generate signal.
        freqs[k] = startFreq + k * freqInc;
        for (unsigned i = 0, t = 0; t < nBlocks; ++t)
        {
            float* timeData = (float*)(timeSeries->timeSeriesData(t, 0, 0));
            double time, arg;
            time = double(i++) / sampleRate;
            arg = 2.0 * math::pi * freqs[k] * time;
            for (unsigned c = 0; c < _nChannels * 2; ++c)
            {
                timeData[c] = cos(arg) + _counter;
                timeData[++c] = sin(arg) + _counter;
            }
        }
    }
    _readyMade.append( timeSeries );
    return timeSeries;
}

} // namespace lofar
} // namespace pelican
