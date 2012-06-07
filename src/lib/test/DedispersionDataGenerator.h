#ifndef DEDISPERSIONDATAGENERATOR_H
#define DEDISPERSIONDATAGENERATOR_H

#include <QList>
class QMutex;
class QWaitCondition;

/**
 * @file DedispersionDataGenerator.h
 */

namespace pelican {
class DataBlob;

namespace lofar {
class SpectrumDataSetStokes;
class DedispersionSpectra;

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

        /// convenience method to clean up generated DedispersionSpectra objects
        static void deleteData( DedispersionSpectra* data );

        /// convenience method to make a deep copy of a data set
        static QList<SpectrumDataSetStokes*> deepCopy(
                const QList<SpectrumDataSetStokes*>& input );

        /// compare two datasets for euality
        static bool equal(
                const QList<SpectrumDataSetStokes*>& input,
                const QList<SpectrumDataSetStokes*>& ref );

        /// fill each block with the specified number of samples
        void setTimeSamplesPerBlock( unsigned num ) { nSamples = num; }

        /// create a dedispersion object (processdd by the dedispersion module)
        DedispersionSpectra* dedispersionData( float dedispersionMeasure );


    protected:
        void copyData( DataBlob* in, DedispersionSpectra* out ) const;
        void wakeUp( QWaitCondition* waiter, QMutex* mutex );

    protected:
        unsigned nSamples; // number of samples per DataBlob
        unsigned nSubbands;
        unsigned nChannels; 
        unsigned startBin; 
        unsigned signalWidth;

        double fch1; // frequency of channel 1
        double foff; // frequency delta (assumed to be -ve)
        double tsamp; // time sample length (seconds)
};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONDATAGENERATOR_H 
