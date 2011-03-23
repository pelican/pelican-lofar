#include "BandPassAdapter.h"
#include <QIODevice>
#include <QString>
#include <QVector>
#include "BinMap.h"
#include "BinnedData.h"
#include "BandPass.h"


namespace pelican {
namespace lofar {


/**
 *@details BandPassAdapter 
 */
BandPassAdapter::BandPassAdapter( const ConfigNode& config )
    : AbstractServiceAdapter(config)
{
}

/**
 *@details
 */
BandPassAdapter::~BandPassAdapter()
{
}

void BandPassAdapter::deserialise(QIODevice* device) 
{
    BandPass* blob = (BandPass*) dataBlob();
    // format of data file
    // uint number of channels
    // float start channel frequency
    // float end channel frequency
    // float channel width
    // list of floats corresponding to the paramterised equation (0...nth coefficient)
    QByteArray b = device->readLine();
    unsigned int nChan = b.toUInt();
    float startFreq = (device->readLine()).toFloat();
    float endFreq = (device->readLine()).toFloat();
    float deltaF = (device->readLine()).toFloat();
    float rms = (device->readLine()).toFloat();
    float median = (device->readLine()).toFloat();
    QVector<float> params;
    while( device->canReadLine() ) {
        params.append( (device->readLine()).toFloat() );
    }
    BinMap map( nChan );
    map.setStart(startFreq);
    map.setBinWidth(deltaF);
    blob->setData( map, params );
    blob->setMedian(median);
    blob->setRMS(rms);

}

} // namespace lofar
} // namespace pelican
