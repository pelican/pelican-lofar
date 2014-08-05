#ifndef H5CVAPPLICATION_H
#define H5CVAPPLICATION_H

#include <QString>
#include "TimeSeriesDataSet.h"
#include "AdapterTimeSeriesDataSet.h"

/**
 * @file H5CVApplication.h
 */

namespace pelican {

namespace ampp {

/**
 * @class H5CVApplication
 *  
 * @brief
 *    Class for defining the main runtime and pipelines for their H5CV use case.
 *
 * @details
 * 
 */

class H5CVApplication
{
    public:
	/// Constructor
        H5CVApplication( int argc, char** argv, const QString& streamId  );
	/// Destructor
        ~H5CVApplication();

    private:
};

} // namespace ampp
} // namespace pelican
#endif // H5CVAPPLICATION_H 
