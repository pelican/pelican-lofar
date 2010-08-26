#include "viewer/SpectrumDataSetWidget.h"
#include "lib/SpectrumDataSet.h"

#include <iostream>

using namespace std;

namespace pelican {
namespace lofar {


SpectrumDataSetWidget::SpectrumDataSetWidget(const ConfigNode& config,
        QWidget* parent)
: DataBlobWidget(config, parent), _integrationCount(0)
{
    setupUi(this);
    plot->setXLabel("Frequency Index");
    plot->setYLabel("Amplitude");
    plot->showGrid(true);
}


void SpectrumDataSetWidget::updateData(DataBlob* data)
{
//    cout << "SpectrumDataSetWidget::updateData()" << endl;

    // Get data selection from widget controls.
    unsigned subband = spinBox_subband->value() - 1;
    unsigned polarisation = spinBox_polarisation->value() - 1;
    unsigned timeSample = spinBox_timeBlock->value() - 1;
    unsigned integrationMax = spinBox_integrationCount->value();

    // Extra plot data from the data blob.
    SpectrumDataSetStokes* spectra = (SpectrumDataSetStokes*)data;
    unsigned nTimeBlocks = spectra->nTimeBlocks();
    unsigned nSubbands = spectra->nSubbands();
    unsigned nPolarisations = spectra->nPolarisations();
    unsigned nChannels = spectra->nChannels(0);

    if (subband >= nSubbands || polarisation >= nPolarisations || timeSample >= nTimeBlocks) {
        plot->clear();
        return;
    }

    if (_spectrumAmp.size() != nChannels)
    {
        plot->setTitle(QString("Spectrum "
                "(sample %1/%2, sub-band %3/%4, polarisation %5/%6)")
                .arg(timeSample + 1).arg(nTimeBlocks)
                .arg(subband + 1).arg(nSubbands)
                .arg(polarisation + 1).arg(nPolarisations));
        _plot(_spectrumAmp);
        _spectrumAmp.resize(nChannels);
    }

    float* spectrum = spectra->spectrumData(timeSample, subband, polarisation);

    for (unsigned i = 0u; i < nChannels; ++i)
        _spectrumAmp[i] += (double)spectrum[i];

    if (++_integrationCount >= integrationMax)
    {
        cout << "max=" << integrationMax << " iteration=" <<  _integrationCount << endl;
        plot->setTitle(QString("Spectrum "
                "(sample %1/%2, subband %3/%4, polarisation %5/%6)")
                .arg(timeSample + 1).arg(nTimeBlocks)
                .arg(subband + 1).arg(nSubbands)
                .arg(polarisation + 1).arg(nPolarisations));
        _plot(_spectrumAmp);
        _spectrumAmp.clear();
        _integrationCount = 0;
    }
}


void SpectrumDataSetWidget::_plot(const vector<double>& vec)
{
    unsigned nChannels = vec.size();
    vector<double> frequencyIndex(nChannels);
    vector<double> avvec(nChannels);
    //     std::cout << "t, s, p) = " << timeSample << " " << subband << " " << polarisation << std::endl;
    //     std::cout << "spectrum[0] = " << spectrum[0] << std::endl;


    //cout << "----- spectrum[0] = " << spectrum[0] << " " << spectrumAmp[0] << endl;
    //cout << "----- spectrum[1] = " << spectrum[1] << " " << spectrumAmp[1] << endl;
    // Set the plot title.
    //     plot->setTitle(QString("Spectrum (sample %1/%2, subband %3/%4, polarisation %5/%6)")
    //       .arg(timeSample).arg(nTimeBlocks)
    //       .arg(subband).arg(nSubbands)
    //       .arg(polarisation).arg(nPolarisations));

    for (unsigned i = 0; i < nChannels; ++i) {
        frequencyIndex[i] = double(i);
        avvec[i] = vec[i] / _integrationCount;
    }

    // Update the plot with the spectrum.
    if (nChannels > 0)
        plot->plotCurve(nChannels, &frequencyIndex[0], &avvec[0]);
}


} // namespace lofar
} // namespace pelican
