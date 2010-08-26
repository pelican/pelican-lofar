#include "lib/SpectrumDataSet.h"
#include "pelican/output/PelicanTCPBlobServer.h"
#include "pelican/utility/ConfigNode.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QString>

#include <vector>
#include <cmath>
#include <iostream>

using namespace std;
using namespace pelican;
using namespace pelican::lofar;

int main(int argc, char** argv)
{
    // Construct a Pelican TCP blob server to send data.
    QCoreApplication app(argc, argv);
    QString xml = ""
            "<PelicanTCPBlobServer>"
            "   <connection port=\"6969\"/>"
            "</PelicanTCPBlobServer>";

    ConfigNode config(xml);
    PelicanTCPBlobServer server(config);


    // Construct a spectrum set object to send.
    unsigned nTimeBlocks = 8;
    unsigned nSubbands = 31;
    unsigned nPolarisations = 4;
    unsigned nChannels = 64;
    SpectrumDataSetStokes spectra;
    spectra.resize(nTimeBlocks, nSubbands, nPolarisations, nChannels);

    sleep(1);

    // Start sending spectra.
    unsigned long counter = 0;
    float* data;
    float nPeriods, arg;
    while(true)
    {
        cout << "Sending spectra blob " << counter << endl;

        // Fill spectra with interesting data.
        for (unsigned t = 0; t < nTimeBlocks; ++t) {
            for (unsigned s = 0; s < nSubbands; ++s) {
                for (unsigned p = 0; p < nPolarisations; ++p) {
                    for (unsigned c = 0; c < nChannels; ++c) {

                        data = spectra.spectrumData(t, s, p);
                        nPeriods = float(s + 1) * float(p + 1);
                        arg = 2.0f * M_PI * float(c) * nPeriods / float(nChannels);
                        data[c] = float(t) + (1.0f + float(counter) / 2.0f) * sin(arg);

                    }
                }
            }
        }

        spectra.setVersion(QString::number(counter));
        server.send("SpectrumDataSetStokes", &spectra);
        counter++;

        if (counter > 1e3) counter = 0;
//        sleep(1);
    }
}
