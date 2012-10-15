#ifndef DEDISPERSIONDATAANALYSIS_H
#define DEDISPERSIONDATAANALYSIS_H


#include "pelican/data/DataBlob.h"
#include <QVector>
#include <QPair>
#include "DedispersionEvent.h"

/**
 * @file DedispersionDataAnalysis.h
 */

namespace pelican {

namespace lofar {
class DedispersionSpectra;

/**
 * @class DedispersionDataAnalysis
 *  
 * @brief
 *    DataBlob to contain the analysis results 
 *    of dedispersed data
 * @details
 * 
 */

class DedispersionDataAnalysis : public DataBlob
{
        typedef QList< DedispersionEvent > EventIndexT;

    public:
        DedispersionDataAnalysis(  );
        ~DedispersionDataAnalysis();

        /// add an event
        void addEvent( unsigned dm, unsigned timeBin, float mfBinFactor, float mfBinValue );

        /// return the number of events found
        int eventsFound() const; 

        /// reset the Analysis data and set a different base data layer
        void reset( const DedispersionSpectra* );

        /// return the underlying data object
        inline const DedispersionSpectra* data() const { return _data; };

        /// return a list of events found
        const QList<DedispersionEvent>& events() const;

        /// Return the rms of the data if it has been set;                                                                                                  
        float getRMS() const { return _rms; }

        /// Set the rms of the data;                                                                                                                        
        void setRMS(float rms) { _rms = rms; }

    private:
        const DedispersionSpectra* _data;
        EventIndexT _eventIndex;
        float _rms;
};

PELICAN_DECLARE_DATABLOB(DedispersionDataAnalysis)

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONDATAANALYSIS_H 
