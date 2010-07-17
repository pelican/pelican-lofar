#include "viewer/SubbandSpectrumWidget.h"
#include "lib/SubbandSpectra.h"

#include <iostream>
using namespace std;

namespace pelican {
namespace lofar {


SubbandSpectrumWidget::SubbandSpectrumWidget(QWidget* parent)
: DataBlobWidget(parent)
{
    setupUi(this);
    plot->setXLabel("Frequency Index");
    plot->setYLabel("Amplitude");
    plot->showGrid(true);

}

void SubbandSpectrumWidget::updateData(DataBlob* data)
{
    // Get data selection from widget controls.
     unsigned subband = spinBox_subband->value();
     unsigned polarisation = spinBox_polarisation->value();
     unsigned timeSample = 0;

//     // Extra plot data from the data blob.
//     SubbandSpectraStokes* spectra = (SubbandSpectraStokes*)data;
//     unsigned nTimeBlocks = spectra->nTimeBlocks();
//     unsigned nSubbands = spectra->nSubbands();
//     unsigned nPolarisations = spectra->nPolarisations();
//     unsigned nChannels = spectra->ptr(0,0,0)->nChannels();
//
//     if (subband >= nSubbands || polarisation >= nPolarisations) {
//         plot->clear();
//         return;
//     }
//
//     std::vector<double> frequencyIndex(nChannels);
//     std::vector<double> spectrumAmp(nChannels);
//     float* data = spectra->data(timeSample, subband, polarisation)->ptr();
//     for (unsigned i = 0; i < nChannels; ++i) {
//         frequencyIndex[i] = double(i);
//         spectrumAmp[i] = data[i];
//     }
//
//     //cout << "----- spectrum[0] = " << spectrum[0] << " " << spectrumAmp[0] << endl;
//     //cout << "----- spectrum[1] = " << spectrum[1] << " " << spectrumAmp[1] << endl;
//     // Set the plot title.
//     plot->setTitle(QString("Spectrum (subband %1/%2, polarisation %3/%4)")
//             .arg(subband).arg(nSubbands)
//             .arg(polarisation).arg(nPolarisations));
//
//
//     // Update the plot with the spectrum.
//     plot->plotCurve(nChannels, &frequencyIndex[0], &spectrumAmp[0]);
//
//     //cout << "** plotting nChan = " << nChannels << endl;
//     //plot->plotCurve(nChannels, &frequencyIndex[0], &frequencyIndex[0]);
}


} // namespace lofar
} // namespace pelican
