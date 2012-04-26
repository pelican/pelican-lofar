#include "DedispersionAnalyser.h"
#include "DedispersionSpectra.h"
#include "DedispersionDataAnalysis.h"
#include <QVector>


namespace pelican {

namespace lofar {


/**
 *@details DedispersionAnalyser 
 */
DedispersionAnalyser::DedispersionAnalyser( const ConfigNode& config )
    : AbstractModule( config )
{
    
}

DedispersionAnalyser::~DedispersionAnalyser()
{
}

int DedispersionAnalyser::analyse( DedispersionSpectra* data, 
                                    DedispersionDataAnalysis* result ) {

    result->reset(data);

    // Calculate the mean
    double mean = 0.0, stddev = 0.0;
    double total = 0.0;

    QVector<float> dataVector = data->data();
    int vals=dataVector.size();
    for( int j = 0; j < vals; ++j ) {
        total += dataVector[j];
    }
    mean = total/vals;  // Mean for entire array

    // Calculate standard deviation
    total = 0;
    for(int j=0; j < vals; ++j ) {
        total += pow(dataVector[j] - mean, 2);
    }
    stddev = sqrt(total / vals); // Stddev for entire dedispersion array

    // Subtract dm mean from all samples and apply threshold
    int tdms = data->dmBins();
    int nsamp = data->timeSamples();
    for(int dm_count = 0; dm_count < tdms; ++dm_count) {
        for(int j=0; j < nsamp; ++j) {
            total = dataVector[dm_count*nsamp + j] - mean;
            if (abs(total) >= (stddev * 6) ){
                result->addEvent( dm_count, j );
            }
        }
    }
    return result->eventsFound();
}


} // namespace lofar
} // namespace pelican
