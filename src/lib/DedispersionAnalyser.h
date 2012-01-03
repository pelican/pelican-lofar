#ifndef DEDISPERSIONANALYSER_H
#define DEDISPERSIONANALYSER_H


#include "pelican/modules/AbstractModule.h"

/**
 * @file DedispersionAnalyser.h
 */

namespace pelican {
class DataBlob;
namespace lofar {
class AsyncronousJob;

/**
 * @class DedispersionAnalyser
 *
 * @brief
 *
 * @details
 *
 */

class DedispersionAnalyser : public AbstractModule
{
    public:
        DedispersionAnalyser( const ConfigNode& config );
        ~DedispersionAnalyser();
        void run( DataBlob* ) {};

    private:
};

PELICAN_DECLARE_MODULE(DedispersionAnalyser)
} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONANALYSER_H 
