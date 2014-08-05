#include "DedispersionSpectra.h"
#include "SpectrumDataSet.h"


namespace pelican {

namespace ampp {


/**
 *@details DedispersionSpectra 
 */
DedispersionSpectra::DedispersionSpectra()
    : DataBlob("DedispersionSpectra"), _timeBins(0)
{
}

/**
 *@details
 */
DedispersionSpectra::~DedispersionSpectra()
{
}

void DedispersionSpectra::resize( unsigned timebins, unsigned dedispersionBins,
                                  float dedispersionBinStart, float dedispersionBinWidth ) { 
    _timeBins = timebins;
    _dmBin.reset( dedispersionBins );
    _dmBin.setStart( dedispersionBinStart );
    _dmBin.setBinWidth( dedispersionBinWidth );
    _data.resize( timebins*dedispersionBins ); 
}

float DedispersionSpectra::dmAmplitude( unsigned timeSlice, float dm ) const {
    int dm_index = dmIndex(dm);
    return dmAmplitude( timeSlice, dm_index );
}

float DedispersionSpectra::dmAmplitude( unsigned timeSlice, int dm ) const {
    Q_ASSERT( (int)dm < dmBins() );
    int index = timeSlice + _timeBins * dm;
    Q_ASSERT( index < _data.size() );
    return _data[index];
}

int DedispersionSpectra::dmIndex( float dm ) const {
    return _dmBin.binIndex(dm);
}

float DedispersionSpectra::dm( unsigned dm ) const {
    return _dmBin.binAssignmentNumber( dm );
}

void DedispersionSpectra::setInputDataBlobs( const QList<SpectrumDataSetStokes*>& blobs ) {
    _inputBlobs = blobs;
}

void DedispersionSpectra::setFirstSample( unsigned int sampleNumber ) {
    _firstSampleNumber = sampleNumber;
}

double DedispersionSpectra::getTime( unsigned int sampleNumber ) const {
    // N.B not resilient to inhomogenous sample times across blobs
    // to fix this we would need to find the blob corresponding to the
    // sample first.
    return _inputBlobs[0]->getTime( sampleNumber + _firstSampleNumber );
}

} // namespace ampp
} // namespace pelican
