#include <iostream>
#include <QtCore/QCoreApplication>
#include "pelican/emulator/EmulatorDriver.h"
#include "ABEmulator.h"

using namespace pelican;
using namespace pelican::ampp;

int main(int argc, char* argv[])
{
    // Create a QCoreApplication
    QCoreApplication app(argc, argv);

    try {
        ConfigNode emulatorConfig("<ABEmulator>"
                                  "<packet samples=\"1024\" interval=\"100\" />"
                                  "<signal period=\"20\" />"
                                  "<connection host=\"127.0.0.1\" port=\"9999\" />"
                                  "</ABEmulator>");
        EmulatorDriver emulator(new ABEmulator(emulatorConfig));
        return app.exec();
    }

    // Catch any error messages from Pelican
    catch (const QString& err) {
        std::cerr << "Error: " << err.toStdString() << std::endl;
    }

    return 0;
}

