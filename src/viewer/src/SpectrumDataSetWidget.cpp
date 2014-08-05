#include "viewer/SpectrumDataSetWidget.h"
#include "lib/SpectrumDataSet.h"
#include <algorithm>

#include <iostream>

using namespace std;

namespace pelican {
namespace ampp {


SpectrumDataSetWidget::SpectrumDataSetWidget(const ConfigNode& config,
        QWidget* parent)
: DataBlobWidget(config, parent), _integrationCount(1)
{
    setupUi(this);
    plot->setXLabel("Frequency Index");
    plot->setYLabel("Amplitude");
    plot->showGrid(true);

    // make some connections
    // these should only really be active in "paused" mode
    connect( spinBox_subband, SIGNAL( valueChanged(int) ), this, SLOT( doPlot() ) );
    connect( spinBox_polarisation, SIGNAL( valueChanged(int) ), this, SLOT( doPlot() ) );
    connect( spinBox_timeBlock, SIGNAL( valueChanged(int) ), this, SLOT( doPlot() ) );
    connect( spinBox_integrationCount, SIGNAL( valueChanged(int) ), this, SLOT( doPlot() ) );
}


void SpectrumDataSetWidget::updateData(DataBlob* data)
{
    cout << "SpectrumDataSetWidget::updateData()" << endl;

    // Extra plot data from the data blob.
    SpectrumDataSetStokes* spectra = (SpectrumDataSetStokes*)data;
    unsigned nTimeBlocks = spectra->nTimeBlocks();
    unsigned nSubbands = spectra->nSubbands();
    unsigned nPolarisations = spectra->nPolarisations();
    //unsigned nChannels = spectra->nChannels();
    spinBox_subband->setMaximum( nSubbands  );
    spinBox_subband->setMinimum( 1 );
    spinBox_polarisation->setMaximum( nPolarisations );
    spinBox_polarisation->setMinimum( 1 );
    spinBox_timeBlock->setMaximum( nTimeBlocks );
    spinBox_timeBlock->setMinimum( 1 );
    _spectra = spectra;
    doPlot();
}

void SpectrumDataSetWidget::doPlot() {
    unsigned nTimeBlocks = _spectra->nTimeBlocks();
    unsigned nSubbands = _spectra->nSubbands();
    unsigned nPolarisations = _spectra->nPolarisations();
    unsigned nChannels = _spectra->nChannels();

    // Get data selection from widget controls.
    unsigned subband = std::min((unsigned)spinBox_subband->value() , nSubbands) - 1;
    unsigned polarisation = std::min((unsigned) spinBox_polarisation->value() , nPolarisations) - 1;
    unsigned timeSample = std::min((unsigned) spinBox_timeBlock->value() , nTimeBlocks) - 1;
    unsigned integrationMax = spinBox_integrationCount->value();

    plot->setTitle(QString("Spectrum "
                "(sample %1/%2, sub-band %3/%4, polarisation %5/%6)")
            .arg(timeSample + 1).arg(nTimeBlocks)
            .arg(subband+1).arg(nSubbands)
            .arg(polarisation +1).arg(nPolarisations));
    _spectrumAmp.resize(nChannels);

    float* spectrum = _spectra->spectrumData(timeSample, subband, polarisation);

    for (unsigned i = 0u; i < nChannels; ++i) {
        _spectrumAmp[i] = (double)spectrum[i];
        std::cout << "spectrumAmp[" << i << "] = " << _spectrumAmp[i] << std::endl;
    }
    _plot(_spectrumAmp);

/*
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
*/
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


} // namespace ampp
} // namespace pelican
