#ifndef UDPBFAPPLICATION_H
#define UDPBFAPPLICATION_H

#include <QString>
#include "TimeSeriesDataSet.h"
#include "AdapterTimeSeriesDataSet.h"

/**
 * @file UdpBFApplication.h
 */

namespace pelican {

namespace ampp {

/**
 * @class UdpBFApplication
 *  
 * @brief
 *    Class for defining the main runtime and pipelines for their UdpBF use case.
 *
 * @details
 * 
 */

class UdpBFApplication
{
    public:
	/// Constructor
        UdpBFApplication( int argc, char** argv, const QString& streamId  );
	/// Destructor
        ~UdpBFApplication();

    private:
};

} // namespace ampp
} // namespace pelican
#endif // UDPBFAPPLICATION_H 
