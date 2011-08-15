#include "PPFChanneliser.h"
#include "SpectrumDataSet.h"
#include "TimeSeriesDataSet.h"

#include "pelican/utility/ConfigNode.h"

#include <QtCore/QTime>
#include <QtCore/QFile>
#include <QtCore/QTextStream>
#include <QtCore/QFile>

#include <cstdio>
#include <complex>
#include <vector>
#include <cmath>

using namespace pelican;
using namespace pelican::lofar;
using namespace std;
typedef std::complex<float> Complex;

// Forward declaration of function to create a configuration XML node for
// the channeliser.
QString createXMLConfig(unsigned nChannels, unsigned nThreads, unsigned nTaps,
        const QString& windowType = "kaiser");



int main(int /*argc*/, char** /*argv*/)
{
    // ==== PPF channeliser settings.
    unsigned num_subbands = 1;
    unsigned num_pols     = 1;
    unsigned num_threads  = 1;
    unsigned num_channels = 512;
    unsigned num_taps     = 8;

    printf("= Channeliser setup:\n");
    printf("     num_channels = %i\n", num_channels);
    printf("     num_taps     = %i\n", num_taps);

    ConfigNode config(createXMLConfig(num_channels, num_threads, num_taps));
    PPFChanneliser channeliser(config);

    // ==== Profile settings.
    double bandwidth        = (double)num_channels;   // Hz
    double start_freq       = -bandwidth/ 2.0;   // Hz

    unsigned num_freq_steps = 1024.0;

    double freq_inc    = bandwidth / num_freq_steps;
    double end_freq    = start_freq + freq_inc * num_freq_steps;

    double sample_rate = bandwidth * 2;  // Hz

    printf("\n= Scanning freqs:\n");
    printf("     from        : %.3f Hz\n", start_freq);
    printf("     to          : %.3f Hz\n", end_freq);
    printf("     in steps of : %.3f kHz.\n", freq_inc / 1e3);
    printf("\n= Sample rate = %f\n", sample_rate);
    printf("= Freq inc.   = %f\n", freq_inc);

    // ==== Initialise data arrays.
    TimeSeriesDataSetC32 time_series;
    time_series.resize(num_taps, num_subbands, num_pols, num_channels);
    SpectrumDataSetC32 spectra;
    spectra.resize(num_taps, num_subbands, num_pols, num_channels);

    vector<double> freqs(num_freq_steps);

    for (unsigned k = 0; k < num_freq_steps; ++k)
    {

        // Generate a time series for the k'th frequency.
        // The time series needs to be num_taps * num_channels long
        freqs[k] = start_freq + k * freq_inc;
        for (unsigned i = 0, t = 0; t < num_taps; ++t)
        {
            Complex* time_data = time_series.timeSeriesData(t, 0, 0);
            for (unsigned c = 0; c < num_channels; ++c)
            {
                double time = double(i++) / sample_rate;
                double arg = 2.0 * M_PI * freqs[k] * time;
                time_data[c] += Complex(cos(arg), sin(arg));
            }
        }

    }
    time_series.write("temp_time_series.dat");

    // Run the channeliser.
    channeliser.run(&time_series, &spectra);

    spectra.write("temp_spectrum.dat", -1, -1, 4);

//    // ==== Write the channel profile to file.
//    QFile file("channel_profile_2.dat");
//    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
//        return 1;
//    QTextStream out(&file);
//    for (unsigned j = 0; j < num_channels; ++j)
//    {
//        out << j << " ";
//        for (unsigned i = 0; i < num_freq_steps; ++i)
//        {
//            Complex* s = spectra[i].spectrumData(num_taps-1, 0, 0);
//            out << 20 * log10(abs(s[j])) << " ";
//        }
//        out << endl;
//    }
//    file.close();

    return 0;
}




QString createXMLConfig(unsigned nChannels, unsigned nThreads, unsigned nTaps,
        const QString& windowType)
{
    QString xml =
            "<PPFChanneliser>"
            "   <outputChannelsPerSubband value=\"" + QString::number(nChannels) + "\"/>"
            "   <processingThreads value=\"" + QString::number(nThreads) + "\"/>"
            "   <filter nTaps=\"" + QString::number(nTaps) + "\" filterWindow=\"" + windowType + "\"/>"
            "</PPFChanneliser>";
    return xml;
}
