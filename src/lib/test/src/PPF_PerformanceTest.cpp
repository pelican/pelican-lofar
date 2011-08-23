#include "PPFChanneliser.h"
#include "SpectrumDataSet.h"
#include "TimeSeriesDataSet.h"

#include "pelican/utility/ConfigNode.h"

#include <QtCore/QTime>
#include <QtCore/QFile>
#include <QtCore/QTextStream>
#include <iostream>
#include <complex>
#include <vector>
#include <cmath>

using std::cout;
using std::endl;
namespace P = pelican;
namespace PL = pelican::lofar;

// Forward declaration of function to create a configuration XML node for
// the channeliser.
QString createXMLConfig(unsigned nChannels, unsigned nThreads, unsigned nTaps,
        const QString& windowType = "kaiser");


int main(int /*argc*/, char** /*argv*/)
{
    bool verbose = true;
    unsigned nChannels = 16;
    unsigned nSubbands = 62;
    unsigned nPols = 2;
    unsigned nTaps = 8;

    unsigned timesPerChunk =  512 * 1000;

    if (timesPerChunk%nChannels)
    {
        cout << "Setup error" << endl;
        return 1;
    }

    unsigned _nBlocks = timesPerChunk / nChannels;

    cout << endl << "***** testing PPFChanneliser run() method ***** " << endl;
    cout << endl;
    cout << "-------------------------------------------------" << endl;
    cout << "[PPFChanneliser]: run() " << endl;
    cout << "- nChan = " << nChannels << endl;
    if (verbose) {
        cout << "- nTaps = " << nTaps << endl;
        cout << "- nBlocks = " << _nBlocks << endl;
        cout << "- nSubbands = " << nSubbands << endl;
        cout << "- nPols = " << nPols << endl;
    }

    // Setup the channeliser.
    unsigned nThreads = 2;
    P::ConfigNode config(createXMLConfig(nChannels, nThreads, nTaps));

    PL::PPFChanneliser channeliser(config);
    PL::SpectrumDataSetC32 spectra;

    unsigned nIter = 5;
    // Run once to size up buffers etc.
    for (unsigned i = 0; i < nIter; ++i)
    {
        PL::TimeSeriesDataSetC32 timeSeries;
        timeSeries.resize(_nBlocks, nSubbands, nPols, nChannels);
        QTime timer;
        timer.start();
        channeliser.run(&timeSeries, &spectra);
        int elapsed = timer.elapsed();
        cout << "* Run [" << i << "] time = " << elapsed << " ms. [" << nThreads << " threads]" << endl;
    }


    cout << endl;
    cout << " (data time = " << _nBlocks * nChannels * 5e-3 << " ms.)" << endl;
    cout << "-------------------------------------------------" << endl;

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
