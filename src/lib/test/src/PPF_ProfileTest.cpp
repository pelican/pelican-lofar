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


//------------------------------------------------------------------------------
class PPF_test
{
    public:
        PPF_test()
        {
            num_subbands = 1;
            num_pols     = 1;
            num_threads  = 1;
            num_channels = 16;
            num_taps     = 8;
            window_type  = "kaiser";
        }

        // Generate a set of channel profiles.
        void test_channel_profiles();

        void populate_time_series(unsigned id, double freq, double delta_t);

        // Create an XML config node require to setup the channeliser.
        ConfigNode createXMLConfig();

    public:
        unsigned num_subbands;
        unsigned num_pols;
        unsigned num_threads;
        unsigned num_channels;
        unsigned num_taps;
        QString window_type;

        vector<TimeSeriesDataSetC32> time_series;
        vector<SpectrumDataSetC32> spectra;

}; // class PPF_test
//------------------------------------------------------------------------------



int main(int /*argc*/, char** /*argv*/)
{
    PPF_test t;
    t.num_channels = 16;
    t.num_taps     = 8;
    t.test_channel_profiles();

    return 0;
}




//==============================================================================
//==============================================================================



void PPF_test::test_channel_profiles()
{
    PPFChanneliser channeliser(createXMLConfig());

    // Profile settings. FIXME check these....
    double bandwidth   = (double)num_channels;   // Hz
    double start_freq  = -(bandwidth / 2.0);   // Hz
    unsigned num_freq_steps = 512.0;
    double freq_inc    = bandwidth / num_freq_steps;
    double end_freq    = start_freq + freq_inc * num_freq_steps;
    double sample_rate = bandwidth;  // Hz

    // Setup input and output data arrays.
    time_series.resize(1);
    time_series[0].resize(num_taps, num_subbands, num_pols, num_channels);
    spectra.resize(num_freq_steps);
    for (unsigned i = 0; i < num_freq_steps; ++i)
        spectra[i].resize(num_taps, num_subbands, num_pols, num_channels);
    vector<double> freqs(num_freq_steps);

    // Generate channel profile by scanning though frequencies.
    for (unsigned k = 0; k < num_freq_steps; ++k)
    {

        // Generate a time series for the k'th frequency.
        // The time series needs to be num_taps * num_channels long
        freqs[k] = start_freq + k * freq_inc;
        for (unsigned i = 0, t = 0; t < num_taps; ++t)
        {
            Complex* time_data = time_series[0].timeSeriesData(t, 0, 0);
            for (unsigned c = 0; c < num_channels; ++c)
            {
                double time = double(i++) / sample_rate;
                double arg = 2.0 * M_PI * freqs[k] * time;
                time_data[c] = Complex(cos(arg), sin(arg));
            }
        }

        // Run the channeliser.
        channeliser.run(&time_series[0], &spectra[k]);
    }


    // ==== Write the channel profile to file.
    QFile file("channel_profiles.dat");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        exit(1);
    QTextStream out(&file);
    for (unsigned i = 0; i < num_freq_steps; ++i)
    {
        Complex* s = spectra[i].spectrumData(num_taps-1, 0, 0);
        out << freqs[i] << " ";
        for (unsigned p = 0; p < num_channels; ++p)
        {
            // NOTE: there are n taps spectra generated per call to the channeliser.
            // this just prints the first one.
            out << 20 * log10(abs(s[p])) << " ";
        }
        out << endl;
    }
    file.close();
}


void PPF_test::populate_time_series(unsigned id, double freq, double delta_t)
{
    for (unsigned b = 0; b < num_taps; ++b)
    {
        Complex* data = time_series[id].timeSeriesData(b, 0, 0);

        for (unsigned t = 0; t < num_channels; ++t)
        {
            double time = double(b * num_channels + t) * delta_t;
            double arg = 2.0 * M_PI * freq * time;
            data[t] = Complex(cos(arg), sin(arg));
        }
    }
}


ConfigNode PPF_test::createXMLConfig()
{
    QString xml =
            "<PPFChanneliser>"
            "   <outputChannelsPerSubband value=\"" + QString::number(num_channels) + "\"/>"
            "   <processingThreads value=\"" + QString::number(num_threads) + "\"/>"
            "   <filter num_taps=\"" + QString::number(num_taps) + "\" filterWindow=\"" + window_type + "\"/>"
            "</PPFChanneliser>";

    return ConfigNode(xml);
}

