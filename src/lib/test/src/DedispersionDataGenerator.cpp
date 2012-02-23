#include "DedispersionDataGenerator.h"
#include "pelican/utility/ConfigNode.h"
#include "pelican/output/DataBlobFile.h"
#include "SpectrumDataSet.h"
#include <fstream>


namespace pelican {

namespace lofar {


/**
 *@details DedispersionDataGenerator 
 */
DedispersionDataGenerator::DedispersionDataGenerator()
{
    // set default values (typical LOFAR station values)
    nSamples = 16; // samples per blob
    nSubbands = 32;
    nChannels = 64; // 2048 total channels (32x64)

    fch1 = 150;
    foff = -6.0/(double)(nSubbands*nChannels);
    tsamp = 0.00032768; // time sample length (seconds)
}

/**
 *@details
 */
DedispersionDataGenerator::~DedispersionDataGenerator()
{
}

QList<SpectrumDataSetStokes*> DedispersionDataGenerator::generate( int numberOfBlocks , float dm )
{
    QList<SpectrumDataSetStokes*> data;

    for( int i=0; i < numberOfBlocks; ++i ) {
        SpectrumDataSetStokes* stokes = new SpectrumDataSetStokes;
        stokes->resize(nSamples, nSubbands, 1, nChannels);
        data.append(stokes);

        int offset = i * nSamples;
        //stokes->setLofarTimestamp(channeliserOutput->getLofarTimestamp());
        for (unsigned int t = 0; t < nSamples; ++t ) {
            for (unsigned s = 0; s < nSubbands; ++s ) {
                for (unsigned c = 0; c < nChannels; ++c) {
                    int absChannel = s * nChannels + c;
                    int index = (int)( dm * (4148.741601 * ((1.0 / (fch1 + (foff * absChannel)) /
                        (fch1 + (foff * absChannel))) - (1.0 / fch1 / fch1))/tsamp ) );
                    int sampleNumber = index - offset;

                    float* I = stokes->spectrumData(t, s, 0);
                    // add a signal of bandwidth 10
                    if( (int)t >= sampleNumber && (int)t < sampleNumber + 10 ) {
                        I[c] = 1.0;
                    } else {
                        I[c] = 0.0;
                    }
                }
            }
        }
    }
    return data;
}

void DedispersionDataGenerator::writeToFile( const QString& filename, const QList<SpectrumDataSetStokes*>& data ) {
    ConfigNode dummy;
    DataBlobFile writer(dummy);
    writer.addFile(filename ,DataBlobFileType::Homogeneous);
    foreach( const SpectrumDataSetStokes* d, data ) {
        writer.send( QString("input"), d );
    }

}

void DedispersionDataGenerator::deleteData( QList<SpectrumDataSetStokes*>& data )
{
    foreach( SpectrumDataSetStokes* d, data ) {
        delete d;
    }
}

} // namespace lofar
} // namespace pelican
