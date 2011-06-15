#ifndef UDPBFAPPLICATION_H
#define UDPBFAPPLICATION_H

#include <QString>

/**
 * @file UdpBFApplication.h
 */

namespace pelican {

namespace lofar {

/**
 * @class UdpBFApplication
 *  
 * @brief
 *    class for defining the main runtime and pipelines for ther udpBf use case
 * @details
 * 
 */

class UdpBFApplication
{
    public:
        UdpBFApplication( int argc, char** argv, const QString& streamId  );
        ~UdpBFApplication();

    private:
};

} // namespace lofar
} // namespace pelican
#endif // UDPBFAPPLICATION_H 
