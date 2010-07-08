#include "viewer/SubbandSpectrumWidget.h"
#include "lib/ChannelisedStreamData.h"

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

     // Extra plot data from the data blob.
     ChannelisedStreamData* spectra = (ChannelisedStreamData*)data;
     unsigned nChannels = spectra->nChannels();
     unsigned nSubbands = spectra->nSubbands();
     unsigned nPolarisations = spectra->nPolarisations();

     if (subband >= nSubbands || polarisation >= nPolarisations) {
         plot->clear();
         return;
     }

     std::vector<double> frequencyIndex(nChannels);
     std::complex<double>* spectrum = spectra->data(subband, polarisation);
     std::vector<double> spectrumAmp(nChannels);
     for (unsigned i = 0; i < nChannels; ++i) {
         frequencyIndex[i] = double(i);
         spectrumAmp[i] = spectrum[i].real();
     }

     //cout << "----- spectrum[0] = " << spectrum[0] << " " << spectrumAmp[0] << endl;
     //cout << "----- spectrum[1] = " << spectrum[1] << " " << spectrumAmp[1] << endl;
     // Set the plot title.
     plot->setTitle(QString("Spectrum (subband %1/%2, polarisation %3/%4)")
             .arg(subband).arg(nSubbands)
             .arg(polarisation).arg(nPolarisations));


     // Update the plot with the spectrum.
     plot->plotCurve(nChannels, &frequencyIndex[0], &spectrumAmp[0]);

     //cout << "** plotting nChan = " << nChannels << endl;
     //plot->plotCurve(nChannels, &frequencyIndex[0], &frequencyIndex[0]);
}


} // namespace lofar
} // namespace pelican
