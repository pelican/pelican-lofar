#include "BandPassAdapter.h"
#include <QIODevice>
#include <QString>
#include <QVector>
#include "BinMap.h"
#include "BandPass.h"
#include <iostream>


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
    int max = 800; // max number of chars per line
    BandPass* blob = (BandPass*) dataBlob();
    // format of data file
    // # comment
    // uint number of channels
    // float start channel frequency
    // float end channel frequency
    // float channel width
    // list of floats corresponding to the paramterised equation (0...nth coefficient)
    QVector<float> params;
    char c;
    
    int count =0 ,line = 0;
    while( ! device->atEnd() ) {
        ++line;
        device->getChar( &c );
        if( c == '#' ) {
            // dump any comments
            device->readLine(max);
            continue;
        }
        ++count;
        device->ungetChar(c);
        QByteArray b = device->readLine(max).trimmed();
        bool ok;
        params.append( b.toDouble(&ok) );
        if( ! ok ) 
            throw QString("BandPassAdapter: syntax error on line %1").arg(line);
    }

    if( count < 6 ) 
        throw QString("BandPassAdapter: data format wrong : got %1 parameters, expecting at least 6").arg(count);

    unsigned int nChan = (unsigned int)(params[0]);
    float startFreq = params[1];
    float endFreq = params[2];
    float deltaF = params[3];
    float rms = params[4];
    float median = params[5];
    params.remove(0,6);
 
    BinMap map( nChan );
    map.setStart(startFreq);
    map.setBinWidth(deltaF);
    blob->setData( map, params );
    blob->setMedian(median);
    blob->setRMS(rms);

}

} // namespace lofar
} // namespace pelican
