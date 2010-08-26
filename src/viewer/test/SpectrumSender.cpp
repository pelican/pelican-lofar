#include "lib/SpectrumDataSet.h"
#include "pelican/output/PelicanTCPBlobServer.h"
#include "pelican/utility/ConfigNode.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QString>

#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
    unsigned nTimeBlocks = 8;
    unsigned nSubbands = 31;
    unsigned nPolarisations = 4;
    unsigned nChannels = 64;
    QCoreApplication app(argc, argv);

    QString xml = ""
            "<PelicanTCPBlobServer>"
            "   <connection port=\"6969\"/>"
            "</PelicanTCPBlobServer>";
    pelican::ConfigNode config(xml);
    pelican::PelicanTCPBlobServer server(config);
    sleep(1);
    pelican::lofar::SpectrumDataSetStokes spectra;
    spectra.resize(nTimeBlocks, nSubbands, nPolarisations, nChannels);
    unsigned long counter = 0;

    while(true) {
        std::cout << "sending spectra blob " << counter << std::endl;
        float* data;

        // Fill spectra with interesting data.
        for (unsigned t = 0; t < nTimeBlocks; ++t) {
            for (unsigned s = 0; s < nSubbands; ++s) {
                for (unsigned p = 0; p < nPolarisations; ++p) {
                    for (unsigned c = 0; c < nChannels; ++c) {
                        data = spectra.spectrumData(t, s, p);
                        float nPeriods = float(s + 1) * float(p+1);
                        float x = float(c);
                        float arg = 2.0f * float(M_PI) * x * nPeriods / float(nChannels);
                        float amp = (1.0f + float(counter) / 2.0f) * sin(arg);
                        data[c] = amp + float(t);
                    }
                }
            }
        }
//        cout << "*** data[0] = " << data[0] << endl;
//        cout << "*** data[1] = " << data[1] << endl;

        spectra.setVersion(QString::number(counter));
        server.send("SubbandSpectraStokes", &spectra);
        counter++;
        if (counter > 1e3) counter = 0;
//        sleep(1);
    }

    //return 0;
}
