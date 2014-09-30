#include <iostream>
#include <QtCore/QCoreApplication>
#include "pelican/emulator/EmulatorDriver.h"
#include "JTestEmulator.h"

using namespace pelican;
using namespace pelican::ampp;

int main(int argc, char* argv[])
{
    // Create a QCoreApplication
    QCoreApplication app(argc, argv);

    try {
        ConfigNode emulatorConfig("<JTestEmulator>"
                                  "<packet samples=\"256\" interval=\"2560\" />"
                                  "<signal period=\"20\" />"
                                  "<connection host=\"127.0.0.1\" port=\"2001\" />"
                                  "</JTestEmulator>");
        EmulatorDriver emulator(new JTestEmulator(emulatorConfig));
        return app.exec();
    }

    // Catch any error messages from Pelican
    catch (const QString& err) {
        std::cerr << "Error: " << err.toStdString() << std::endl;
    }

    return 0;
}

