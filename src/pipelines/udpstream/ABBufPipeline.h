#ifndef ABBUFPIPELINE_H
#define ABBUFPIPELINE_H

#include "pelican/core/AbstractPipeline.h"
#include "RFI_Clipper.h"
#include "DedispersionModule.h"
#include "DedispersionAnalyser.h"
#include "WeightedSpectrumDataSet.h"
#include "StokesIntegrator.h"
#include "timer.h"

namespace pelican {
namespace ampp {

//class DedispersionModule;

class ABBufPipeline : public AbstractPipeline
{
    public:
        // Constructor.
        ABBufPipeline(const QString& streamIdentifier);

        // Destructor
        ~ABBufPipeline();

        // Initialises the pipeline.
        void init();

        // Defines one iteration of the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

        /// called internally to free up DataBlobs after they are finished with
        void updateBufferLock( const QList<DataBlob*>& );

    protected:
        void dedispersionAnalysis( DataBlob* data );

    private:
        QString _streamIdentifier;

        // Module pointers.
        RFI_Clipper* _rfiClipper;
        DedispersionModule* _dedispersionModule;
        DedispersionAnalyser* _dedispersionAnalyser;
        StokesIntegrator* _stokesIntegrator;

        // Local data blob pointers.
        QList<SpectrumDataSetStokes*> _stokesData;
        LockingPtrContainer<SpectrumDataSetStokes>* _stokesBuffer;
        WeightedSpectrumDataSet* _weightedIntStokes;

        SpectrumDataSetStokes *_stokes;
        SpectrumDataSetStokes *_intStokes;

        unsigned long _counter;
        unsigned _iteration;
        unsigned int _minEventsFound;
        unsigned int _maxEventsFound;

        TimerData _rfiClipperTime;
        TimerData _dedispersionTime;
        TimerData _totalTime;

};

} // namespace ampp
} // namespace pelican

#endif // ABBUFPIPELINE_H

