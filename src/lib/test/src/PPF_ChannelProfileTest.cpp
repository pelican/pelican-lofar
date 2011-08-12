#include "test/PPFChanneliserTest.h"

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
    unsigned num_channels = 16;
    unsigned num_taps     = 8;

    printf("= Channeliser setup:\n");
    printf("     num_channels = %i\n", num_channels);
    printf("     num_taps     = %i\n", num_taps);

    ConfigNode config(createXMLConfig(num_channels, num_threads, num_taps));
    PPFChanneliser channeliser(config);

    // ==== Profile settings.
    double start_freq       = 5.0e6;   // Hz
    double bandwidth        = 1e6;   // Hz
    unsigned num_freq_steps = 1000;    // Number of frequency points scanned in profile.
                                       // this controls how smooth the profile will look.
    double freq_inc         = bandwidth / num_freq_steps;
    double sample_rate      = bandwidth * 2.0;  // Hz

    double end_freq         = start_freq + freq_inc * num_freq_steps;

    printf("\n= Scanning freqs:\n");
    printf("     from        : %.3e Hz\n", start_freq);
    printf("     to          : %.3e Hz\n", end_freq);
    printf("     in steps of : %.3f kHz.\n", freq_inc / 1e3);

    // ==== Initialise data arrays.
    TimeSeriesDataSetC32 data;
    data.resize(num_taps, num_subbands, num_pols, num_channels);
    SpectrumDataSetC32 spectra;
    spectra.resize(num_taps, num_subbands, num_pols, num_channels);
    vector<double> freqs(num_freq_steps);
    vector<vector<Complex> > channel_profile;
    channel_profile.resize(num_channels);
    for (unsigned i = 0; i < num_channels; ++i)
        channel_profile[i].resize(num_freq_steps);


    // ==== Generate channel profile by scanning though frequencies.
    for (unsigned k = 0; k < num_freq_steps; ++k)
    {
        // Generate a time series for the k'th frequency.
        freqs[k] = start_freq + k * freq_inc;

        for (unsigned i = 0, t = 0; t < num_taps; ++t)
        {
            Complex * time_data = data.timeSeriesData(t, 0, 0);
            for (unsigned c = 0; c < num_channels; ++c)
            {
                double time = double(i++) / sample_rate;
                //printf("t = %e\n", time);
                double arg = 2.0 * M_PI * freqs[k] * time;
                time_data[c] = Complex(cos(arg), sin(arg));
            }
        }

        // Run the channeliser.
        channeliser.run(&data, &spectra);

        // Record the amplitude response in each channel.
        Complex* spectrum = spectra.spectrumData(num_taps-1, 0, 0);
        for (unsigned c = 0; c < num_channels; ++c)
        {
            channel_profile[c][k] = spectrum[c];
        }
    }


    // ==== Write the channel profile to file.
    QFile file("channel_profile.dat");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        return 1;
    QTextStream out(&file);
    for (unsigned i = 0; i < num_freq_steps; ++i)
    {
        out << freqs[i] << " ";
        for (unsigned p = 0; p < num_channels; ++p)
        {
            out << 20 * log10(abs(channel_profile[p][i])) << " ";
        }
        out << endl;
    }
    file.close();

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
