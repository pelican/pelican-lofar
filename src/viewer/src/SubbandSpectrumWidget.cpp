#include "viewer/SubbandSpectrumWidget.h"
#include "lib/SubbandSpectra.h"

#include <iostream>
using namespace std;

namespace pelican {
namespace lofar {


SubbandSpectrumWidget::SubbandSpectrumWidget(QWidget* parent)
: DataBlobWidget(parent), _integrationCount(0)
{
    setupUi(this);
    plot->setXLabel("Frequency Index");
    plot->setYLabel("Amplitude");
    plot->showGrid(true);
}

void SubbandSpectrumWidget::updateData(DataBlob* data)
{
    // Get data selection from widget controls.
    unsigned subband = spinBox_subband->value() - 1 ;
    unsigned polarisation = spinBox_polarisation->value() - 1;
    unsigned timeSample = spinBox_timeBlock->value() - 1;
    unsigned integrationMax = spinBox_integrationCount->value();

    // Extra plot data from the data blob.
    SubbandSpectraStokes* spectra = (SubbandSpectraStokes*)data;
    unsigned nTimeBlocks = spectra->nTimeBlocks();
    unsigned nSubbands = spectra->nSubbands();
    unsigned nPolarisations = spectra->nPolarisations();
    unsigned nChannels = spectra->ptr(0,0,0)->nChannels();
    //     std::cout << "nTimeBlocks    = " << nTimeBlocks << std::endl;
    //     std::cout << "nSubbands      = " << nSubbands << std::endl;
    //     std::cout << "nPolarisations = " << nPolarisations << std::endl;
    //     std::cout << "nChannels      = " << nChannels << std::endl;

    if (subband >= nSubbands || polarisation >= nPolarisations || timeSample >= nTimeBlocks) {
            plot->clear();
            return;
    }

    if( _spectrumAmp.capacity() != nChannels )
    {
        plot->setTitle(QString("Spectrum (sample %1/%2, subband %3/%4, polarisation %5/%6)")
                    .arg(timeSample + 1).arg(nTimeBlocks)
                    .arg(subband + 1).arg(nSubbands)
                    .arg(polarisation + 1).arg(nPolarisations));
        _plot(_spectrumAmp);
        _spectrumAmp.resize(nChannels);
    }
    float* spectrum = spectra->ptr(timeSample, subband, polarisation)->ptr();
    for (unsigned i = 0; i < nChannels; ++i) {
            _spectrumAmp[i]+=double(spectrum[i]);
    }
    if( _integrationCount++ % integrationMax == 0 ) {
            plot->setTitle(QString("Spectrum (sample %1/%2, subband %3/%4, polarisation %5/%6)")
            .arg(timeSample + 1).arg(nTimeBlocks)
            .arg(subband + 1).arg(nSubbands)
            .arg(polarisation + 1).arg(nPolarisations));
        _plot(_spectrumAmp);
        _spectrumAmp.clear();
        _integrationCount = 0;
    }
}

void SubbandSpectrumWidget::_plot(const std::vector<double>& vec)
{
    unsigned int nChannels = vec.size();
    std::vector<double> frequencyIndex(nChannels);
    std::vector<double> avvec(nChannels);
            //     std::cout << "t, s, p) = " << timeSample << " " << subband << " " << polarisation << std::endl;
            //     std::cout << "spectrum[0] = " << spectrum[0] << std::endl;

    for (unsigned i = 0; i < nChannels; ++i) {
        frequencyIndex[i] = double(i);
        avvec[i] = vec[i]/_integrationCount;
    }

            //cout << "----- spectrum[0] = " << spectrum[0] << " " << spectrumAmp[0] << endl;
            //cout << "----- spectrum[1] = " << spectrum[1] << " " << spectrumAmp[1] << endl;
            // Set the plot title.

            // Update the plot with the spectrum.
    plot->plotCurve(nChannels, &frequencyIndex[0], &avvec[0]);
    //cout << "** plotting nChan = " << nChannels << endl;
    //plot->plotCurve(nChannels, &frequencyIndex[0], &frequencyIndex[0]);
}


} // namespace lofar
} // namespace pelican
