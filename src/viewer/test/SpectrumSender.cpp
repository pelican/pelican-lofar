#include "lib/ChannelisedStreamData.h"
#include "pelican/output/PelicanTCPBlobServer.h"
#include "pelican/utility/ConfigNode.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QString>

#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
using namespace std;

int main(int argc, char** argv)
{
    unsigned nSubbands = 62;
    unsigned nPolarisations = 2;
    unsigned nChannels = 512;
    QCoreApplication app(argc, argv);

    QString xml = "<PelicanTCPBlobServer>"
            "   <connection port=\"2000\"/>"  // 0 = find unused system port
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
        complex<double>* data = spectra.data();
        for (unsigned i = 0, s = 0; s < nSubbands; ++s) {
            for (unsigned p = 0; p < nPolarisations; ++p) {
                for (unsigned c = 0; c < nChannels; ++c) {
                    double nPeriods = double(s + 1) * double(p+1);
                    double x = double(c);
                    double arg = 2 * M_PI * x * nPeriods / double(nChannels);
                    double re = (1.0 + double(counter) / 2.0) * sin(arg);
                    //double re = x + double(counter);
                    double im = 0.0;
                    data[i] = complex<double>(re, im);
                    i++;
                }
            }
        }
        //cout << "*** data[0] = " << data[0] << endl;
        //cout << "*** data[1] = " << data[1] << endl;

        spectra.setVersion(QString::number(counter));
        server.send("ChannelisedStreamData", &spectra);
        counter++;
        if (counter > 1e3) counter = 0;
        sleep(1);
    }

    //return 0;
}
