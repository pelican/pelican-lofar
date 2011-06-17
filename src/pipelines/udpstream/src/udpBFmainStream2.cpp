#include "UdpBFApplication.h"
#include <QString>
#include <iostream>

using namespace pelican;
using namespace pelican::lofar;

int main(int argc, char* argv[])
{
    // Create a QCoreApplication.
    //QCoreApplication app(argc, argv);
    QString stream = "LofarDataStream2";

    try {
        UdpBFApplication(argc, argv, stream);
    }
    catch (const QString& err) {
        std::cerr << "Error caught in updBFmainStream2.cpp: " << err.toStdString() << std::endl;
    }

    return 0;
}
