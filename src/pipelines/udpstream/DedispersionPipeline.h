#ifndef DEDISPERSIONPIPELINE_H
#define DEDISPERSIONPIPELINE_H

#include <QtCore/QList>
#include "pelican/core/AbstractPipeline.h"
#include "pelican/utility/LockingCircularBuffer.hpp"
#include "LockingPtrContainer.hpp"
#include "PPFChanneliser.h"
#include "StokesGenerator.h"
#include "RFI_Clipper.h"
#include "StokesIntegrator.h"
#include "AdapterTimeSeriesDataSet.h"
#include "TimeSeriesDataSet.h"
#include "SpectrumDataSet.h"
#include "SigprocStokesWriter.h"
#include "DedispersionModule.h"
#include "DedispersionSpectra.h"
#include "DedispersionAnalyser.h"
#include "DedispersionDataAnalysisOutput.h"


/**
 * @file DedispersionPipeline.h
 */

namespace pelican {

namespace lofar {

/**
 * @class DedispersionPipeline
 *  
 * @brief
 *     A dedispersion pipeline for streaming TimeSeries bemaformed Data
 * @details
 * 
 */

class DedispersionPipeline : public AbstractPipeline
{
    public:
        DedispersionPipeline( const QString& streamIdentifier );
        ~DedispersionPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

        /// called internally to free up DataBlobs after they are finished with
        void updateBufferLock( const QList<DataBlob*>& );

    protected:
        void dedispersionAnalysis( DataBlob* data );

    private:
        QString _streamIdentifier;

        /// Module pointers
        PPFChanneliser* _ppfChanneliser;
        StokesGenerator* _stokesGenerator;
        StokesIntegrator* _stokesIntegrator;
        RFI_Clipper* _rfiClipper;
        DedispersionModule* _dedispersionModule;
        DedispersionAnalyser* _dedispersionAnalyser;

        /// Local data blobs
        SpectrumDataSetC32* _spectra;
        QList<SpectrumDataSetC32*> _spectraBuffer;
        QList<DedispersionSpectra*> _dedispersedData;
        LockingPtrContainer<DedispersionSpectra>* _dedispersedDataBuffer;
        TimeSeriesDataSetC32* timeSeries;
        QList<SpectrumDataSetStokes*> _stokesData;
        LockingPtrContainer<SpectrumDataSetStokes>* _stokesBuffer;
        QList<WeightedSpectrumDataSet*> _weightedData;
        LockingPtrContainer<WeightedSpectrumDataSet>* _weightedDataBuffer;

};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONPIPELINE_H 
