#include <QtCore/QCoreApplication>

#include "pelican/emulator/EmulatorDriver.h"
#include "LofarEmulatorDataSim.h"

using namespace pelican;
using namespace pelican::lofar;

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);

    unsigned interval = 500; // microseconds
    unsigned startDelay = 1; // seconds

    unsigned nPols = 2;
    unsigned nSubbands = 61;

    QString xml =
            "<LofarEmulatorDataSim>"
            ""
            "    <connection host=\"127.0.0.1\" port=\"8090\"/>"
            ""
            "    <packetSendInterval value=\"%1\"/>"
            "    <packetStartDelay   value=\"%2\"/>"
            ""
            "    <subbandsPerPacket  value=\"%3\"/>"
            "    <polsPerPacket      value=\"%4\"/>"
            ""
            "</LofarEmulatorDataSim>";

    xml = xml.arg(interval);
    xml = xml.arg(startDelay);
    xml = xml.arg(nSubbands);
    xml = xml.arg(nPols);

    ConfigNode config(xml);
    EmulatorDriver emulator(new LofarEmulatorDataSim(config));

    return app.exec();
}
