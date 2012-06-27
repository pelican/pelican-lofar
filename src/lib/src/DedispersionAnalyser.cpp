#include "DedispersionAnalyser.h"
#include "DedispersionSpectra.h"
#include "DedispersionDataAnalysis.h"
#include "SpectrumDataSet.h"
#include <QVector>


namespace pelican {

namespace lofar {


/**
 *@details DedispersionAnalyser 
 */
DedispersionAnalyser::DedispersionAnalyser( const ConfigNode& config )
    : AbstractModule( config )
{
    // Get configuration options                                                                                                                      
    //unsigned int nChannels = config.getOption("outputChannelsPerSubband", "value", "512").toUInt();                                                 
    _detectionThreshold = config.getOption("detectionThreshold", "in_sigma", "6.0").toFloat();
    _useStokesStats = config.getOption("useStokesStats", "0_or_1").toUInt();
}

DedispersionAnalyser::~DedispersionAnalyser()
{
}

int DedispersionAnalyser::analyse( DedispersionSpectra* data, 
                                    DedispersionDataAnalysis* result ) {

    result->reset(data);
    const QList<SpectrumDataSetStokes* >& d = data->inputDataBlobs(); 
    int nChannels = d[0]->nChannels();
    int nSubbands = d[0]->nSubbands();

    float rms = std::sqrt((float)nChannels*(float)nSubbands);

    std::cout << "---------------" << nChannels << " " << nSubbands << std::endl;
    // Calculate the mean
    double mean = 0.0, stddev = 0.0;
    double total = 0.0;

    QVector<float> dataVector = data->data();
    int vals=dataVector.size();
    /*
    for( int j = 0; j < vals; ++j ) {
        total += dataVector[j];
    }
    mean = total/vals;  // Mean for entire array
    std::cout << "Mean: " << mean << std::endl;

    // Calculate standard deviation
    total = 0;
    for(int j=0; j < vals; ++j ) {
        total += pow(dataVector[j] - mean, 2);
    }
    stddev = sqrt(total / vals); // Stddev for entire dedispersion array
    std::cout << "Stddev: " << stddev << std::endl;
    // Subtract dm mean from all samples and apply threshold
    */
    int tdms = data->dmBins();
    int nsamp = data->timeSamples();
    for(int dm_count = 0; dm_count < tdms; ++dm_count) {
        for(int j=0; j < nsamp; ++j) {
            total = dataVector[dm_count*nsamp + j];
            if (total >= _detectionThreshold * rms){
                result->addEvent( dm_count, j );
            }
        }
    }
    std::cout << "Found " << result->eventsFound() << " events" << std::endl;
    return result->eventsFound();
}


} // namespace lofar
} // namespace pelican
