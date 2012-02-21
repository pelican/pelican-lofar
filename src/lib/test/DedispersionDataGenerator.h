#ifndef DEDISPERSIONDATAGENERATOR_H
#define DEDISPERSIONDATAGENERATOR_H

#include <QList>

/**
 * @file DedispersionDataGenerator.h
 */

namespace pelican {

namespace lofar {
class SpectrumDataSetStokes;

/**
 * @class DedispersionDataGenerator
 *  
 * @brief
 *    Generate test dedispersed signal data
 * @details
 * 
 */

class DedispersionDataGenerator
{
    public:
        DedispersionDataGenerator(  );
        ~DedispersionDataGenerator();

        /// generate a dataset with a single dedispersion signal embedded
        //  The caller is the owner of the resulting data (seed deleteData)
        QList<SpectrumDataSetStokes*> generate( int numberOfBlocks , float dedispersionMeasure );

        /// return the duration (in seconds) of each time sample
        double timeOfSample() const { return tsamp; };

        /// return the value of the starting Frequency
        double startFrequency() const { return fch1; };

        /// return the value of the starting Frequency
        double bandwidthOfSample() const { return foff; };

        /// write a data set to the standard pelican Datablob output file format
        void writeToFile( const QString& filename, const QList<SpectrumDataSetStokes*>& data );

        /// convenience method to clean up the memory of a generated dataset
        static void deleteData( QList<SpectrumDataSetStokes*>& data );

        /// fill each block with the specified number of samples
        void setTimeSamplesPerBlock( unsigned num ) { nSamples = num; }

    protected:
        unsigned nSamples; // number of samples per DataBlob
        unsigned nSubbands;
        unsigned nChannels; 

        double fch1; // frequency of channel 1
        double foff; // frequency delta (assumed to be -ve)
        double tsamp; // time sample length (seconds)
};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONDATAGENERATOR_H 
