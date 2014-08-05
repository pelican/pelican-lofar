#include "H5CVApplication.h"
#include <QString>
#include <iostream>

using namespace pelican;
using namespace pelican::ampp;

int main(int argc, char* argv[])
{
    // Create a QCoreApplication.
    //QCoreApplication app(argc, argv);
    QString stream = "LofarTimeStream2";

    try {
        H5CVApplication(argc, argv, stream);
    }
    catch (const QString& err) {
        std::cerr << "Error caught in h5CVmainStream2.cpp: " << err.toStdString() << std::endl;
    }

    return 0;
}
