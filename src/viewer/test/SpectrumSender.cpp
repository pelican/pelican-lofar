#include "lib/ChannelisedStreamData.h"
#include "pelican/output/PelicanTCPBlobServer.h"
#include "pelican/utility/ConfigNode.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QString>

#include <complex>
#include <vector>
#include <cmath>
#include <iostream>

int main(int argc, char** argv)
{
    unsigned nSubbands = 62;
    unsigned nPolarisations = 2;
    unsigned nChannels = 512;
    QCoreApplication app(argc, argv);

    QString xml = "<PelicanTCPBlobServer>"
            "   <connection port=\"0\"/>"  // 0 = find unused system port
            "</PelicanTCPBlobServer>";
    pelican::ConfigNode config(xml);
    pelican::PelicanTCPBlobServer server(config);
    sleep(1);
    pelican::lofar::ChannelisedStreamData spectra;
    spectra.resize(nSubbands, nPolarisations, nChannels);
    unsigned long counter = 0;

    while(true) {
        std::cout << "sending spectra blob " << counter << std::endl;

        // Fill spectra with interesting data.
        std::complex<double>* data = spectra.data();
        for (unsigned i = 0, s = 0; s < nSubbands; ++s) {
            for (unsigned p = 0; p < nPolarisations; ++p) {
                for (unsigned c = 0; c < nChannels; ++c) {
                    double nPeriods = double(s+p);
                    double x = double(c);
                    double arg = 2 * M_PI * x * nPeriods / double(nChannels);
                    data[i] = std::complex<double>(double(counter) * sin(arg), 0.0);
                    i++;
                }
            }
        }

        spectra.setVersion(QString::number(counter));
        server.send("ChannelisedStreamData", &spectra);
        counter++;
        sleep(1);
    }

    return 0;
}
