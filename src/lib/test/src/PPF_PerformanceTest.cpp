#include "PPFChanneliser.h"
#include "SpectrumDataSet.h"
#include "TimeSeriesDataSet.h"

#include "pelican/utility/ConfigNode.h"

#include <QtCore/QTime>
#include <QtCore/QFile>
#include <QtCore/QTextStream>
#include <cstdio>
#include <cstdlib>
#include <complex>
#include <vector>
#include <cmath>

using namespace pelican;
using namespace pelican::ampp;
using namespace std;

// Prototypes
int run(unsigned num_threads, unsigned num_taps, unsigned num_channels,
        unsigned num_subbands, unsigned num_polarisations, unsigned num_blocks,
        unsigned num_iter);
QString create_xml_config(unsigned nChannels, unsigned nThreads, unsigned nTaps,
        const QString& windowType = "kaiser");


/*
 * TODO:
 *
 * A. Work out number of approx. GFLOPS.
 * B. Work out approx. memory bandwidth.
 *
 * 1. Check memory ordering of TimeSeriesDataSetC32.
 * 2. Check memory ordering of SpectrumDataSetC32.
 * 3. Graph timings for private methods on the channeliser.
 * 4. profile using google-pprof?
 */
int main(int /*argc*/, char** /*argv*/)
{
    // Configuration options.
    unsigned num_times        = 262144; // 2^18
    unsigned num_subbands     = 62;
    unsigned num_channels     = 16;
    unsigned num_polaristions = 2;
    unsigned num_taps         = 4;
    unsigned num_blocks       = num_times / num_channels;

    unsigned num_threads      = 4;
    unsigned num_iter         = 3;

    printf("---------------------------------------------------------------\n");
    printf("- num_times        = %u\n", num_times);
    printf("- data time (ms)   = %f\n", num_times * 5.0e-3);
    printf("- num_subbands     = %u\n", num_subbands);
    printf("- num_channels     = %u\n", num_channels);
    printf("- num_polaristions = %u\n", num_polaristions);
    printf("- num_taps         = %u\n", num_taps);
    printf("- num_blocks       = %u\n", num_blocks);
    printf("- num_iter         = %u\n", num_iter);
    printf("---------------------------------------------------------------\n");

    for (unsigned i = 1; i <= num_threads; ++i)
    {
        int time_taken = run(i, num_taps, num_channels, num_subbands,
                num_polaristions, num_blocks, num_iter);
        printf("[%i threads] time taken = %f ms.\n", i, time_taken / (double)num_iter);
    }

    return EXIT_SUCCESS;
}





int run(unsigned num_threads, unsigned num_taps, unsigned num_channels,
        unsigned num_subbands, unsigned num_polarisations, unsigned num_blocks,
        unsigned num_iter)
{
    // Setup the channeliser.
    ConfigNode config(create_xml_config(num_channels, num_threads, num_taps));

    PPFChanneliser channeliser(config);
    SpectrumDataSetC32 spectra;

    TimeSeriesDataSetC32 time_series;
    time_series.resize(num_blocks, num_subbands, num_polarisations, num_channels);

    // Run once to initilise the buffers etc.
    channeliser.run(&time_series, &spectra);

    // Run num_iter times to time the channeliser.
    QTime timer;
    timer.start();
    for (unsigned i = 0; i < num_iter; ++i)
    {
        channeliser.run(&time_series, &spectra);
    }
    return timer.elapsed();
}


QString create_xml_config(unsigned nChannels, unsigned nThreads, unsigned nTaps,
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
