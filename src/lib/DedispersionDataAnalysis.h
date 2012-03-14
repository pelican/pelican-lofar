#ifndef DEDISPERSIONDATAANALYSIS_H
#define DEDISPERSIONDATAANALYSIS_H


#include "pelican/data/DataBlob.h"
#include <QVector>
#include <QPair>

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
        typedef QVector< QPair<unsigned, unsigned> > EventIndexT;
    public:
        DedispersionDataAnalysis(  );
        ~DedispersionDataAnalysis();

        /// add an event
        void addEvent( unsigned dm, unsigned timeBin );

        /// return the number of events founds
        int eventsFound() const; 

        /// reset the Analysis data and set a different base data layer
        void reset( const DedispersionSpectra* );

    private:
        const DedispersionSpectra* _data;
        EventIndexT _eventIndex;
};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONDATAANALYSIS_H 
