#include "LofarDataBlobGenerator.h"
#include "TimeSeriesDataSet.h"
#include "constants.h"
#include <QDebug>
#include <vector>
#include <complex>
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
    _nSubbands = 62;
    _nChannels = 16;
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
    unsigned timesPerChunk =  262144; // 2^18
    if (timesPerChunk % _nChannels)
        throw QString("Setup error");
    unsigned nBlocks = timesPerChunk / _nChannels;
    timeSeries->resize( nBlocks, _nSubbands, _nPols, _nChannels );

    // Generate channel profile by scanning though frequencies.
    unsigned nSteps   = 1000;     // Number of steps in profile.
    double sampleRate = 50.0e6; // Hz
    double startFreq  = 8.0e6;   // Hz
    double freqInc    = 0.01e6;    // Frequency increment of profile steps.
    std::vector<double> freqs(nSteps);

    typedef std::complex<float> Complex;
    for (unsigned k = 0; k < nSteps; ++k)
    {
        // Generate signal.
        freqs[k] = startFreq + k * freqInc;
        for (unsigned i = 0, t = 0; t < nBlocks; ++t)
        {
            Complex* timeData = timeSeries->timeSeriesData(t, 0, 0);
            double time, arg;
            for (unsigned c = 0; c < _nChannels; ++c)
            {
                time = double(i++) / sampleRate;
                arg = 2.0 * math::pi * freqs[k] * time;
                timeData[c] = Complex(cos(arg), sin(arg));
            }
        }
    }
    return timeSeries;
}

} // namespace lofar
} // namespace pelican
