#include "DedispersionSpectra.h"


namespace pelican {

namespace lofar {


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

} // namespace lofar
} // namespace pelican
