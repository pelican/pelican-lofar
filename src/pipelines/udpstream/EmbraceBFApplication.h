#ifndef EMBRACEBFAPPLICATION_H
#define EMBRACEBFAPPLICATION_H

#include <QString>
#include "TimeSeriesDataSet.h"
#include "AdapterTimeSeriesDataSet.h"

/**
 * @file EmbraceBFApplication.h
 */

namespace pelican {

namespace ampp {

/**
 * @class EmbraceBFApplication
 *  
 * @brief
 *    Class for defining the main runtime and pipelines for their EmbraceBF use case.
 *
 * @details
 * 
 */

class EmbraceBFApplication
{
    public:
	/// Constructor
        EmbraceBFApplication( int argc, char** argv, const QString& streamId  );
	/// Destructor
        ~EmbraceBFApplication();

    private:
};

} // namespace ampp
} // namespace pelican
#endif // EMBRACEBFAPPLICATION_H 
