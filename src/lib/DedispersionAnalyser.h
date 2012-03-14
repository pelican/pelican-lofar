#ifndef DEDISPERSIONANALYSER_H
#define DEDISPERSIONANALYSER_H


#include "pelican/modules/AbstractModule.h"
#include "DedispersionSpectra.h"

/**
 * @file DedispersionAnalyser.h
 */

namespace pelican {
class DataBlob;
namespace lofar {
class DedispersionDataAnalysis;

/**
 * @class DedispersionAnalyser
 *
 * @brief
 *    Extract astronomical events form dedispersion data
 * @details
 *
 */

class DedispersionAnalyser : public AbstractModule
{
    public:
        DedispersionAnalyser( const ConfigNode& config );
        ~DedispersionAnalyser();
        int analyse( DedispersionSpectra*, DedispersionDataAnalysis* );

    private:
};

PELICAN_DECLARE_MODULE(DedispersionAnalyser)
} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONANALYSER_H 
