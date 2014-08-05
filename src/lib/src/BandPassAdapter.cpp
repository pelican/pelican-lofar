#include "BandPassAdapter.h"
#include <QtCore/QIODevice>
#include <QtCore/QMap>
#include <QtCore/QString>
#include <QtCore/QVector>
#include "BinMap.h"
#include "BandPass.h"
#include <iostream>


namespace pelican {
namespace ampp {


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
    QMap<unsigned int,float> killChannels;
    char c;

    int count =0 ,line = 0;
    while( ! device->atEnd() ) {
        bool ok;
        ++line;
        device->getChar( &c );
        switch ( c )
        {
            case '#':
                {
                    // dump any comments
                    device->readLine(max);
                    continue;
                }
            case 'K':
                {
                    QByteArray b = device->readLine(max).simplified();
                    QList<QByteArray> nums = b.split(' ');
                    if( nums.size() < 2 )
                        throw QString("BandPassAdapter: syntax error on line %1: expecting two numbers on Kill line: index frequency").arg(line);
                    unsigned int index = nums[0].toUInt(&ok);
                    if( ! ok )
                        throw QString("BandPassAdapter: syntax error on line %1 - cannot convert index to unsigned integer").arg(line);
                    double value = nums[0].toDouble(&ok);
                    if( ! ok )
                        throw QString("BandPassAdapter: syntax error on line %1 - cannot convert frequency to a double").arg(line);
                    killChannels.insert(index,value);
                    continue;
                }
            default:
                device->ungetChar(c);
        }
        ++count;
        QByteArray b = device->readLine(max).trimmed();
        params.append( b.toDouble(&ok) );
        if( ! ok )
            throw QString("BandPassAdapter: syntax error on line %1").arg(line);
    }

    if( count < 6 )
        throw QString("BandPassAdapter: data format wrong : got %1 parameters, expecting at least 6").arg(count);

    unsigned int nChan = (unsigned int)(params[0]);
    float startFreq = params[1];
    //float endFreq = params[2];
    float deltaF = params[3];
    float rms = params[4];
    float median = params[5];
    params.remove(0,6);

    BinMap map( nChan );
    map.setStart( startFreq );
    map.setBinWidth(deltaF);
    blob->setData( map, params );
    blob->setMedian(median);
    blob->setRMS(rms);
    foreach( unsigned int index, killChannels ) {
        // sanity check, inedx matches frequency specified
        if( map.binIndex(killChannels[index]) != (int)index )
            throw QString("BandPassAdapter:  Frequency kill mismatch: index %1 does not match frequency %2").arg(index).arg(killChannels[index]);
        blob->killChannel(index);
    }
}

} // namespace ampp
} // namespace pelican
