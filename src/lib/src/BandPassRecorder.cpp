#include "BandPassRecorder.h"
#include <QVector>
#include <stdlib.h>
#include <clapack.h>
#include "BandPass.h"
#include "SpectrumDataSet.h"

extern "C" void sgels_(const char*, const int*, const int*,const int*, float*,const int*, float*, int*, float*, int*, int*);

namespace pelican {
namespace lofar {

int BandPassRecorder::sgels(int n, int m, int nrhs, 
                        float *A, int lda, float *B, int ldb, 
                        float *workSpace, int* work )
{
    int info;
    sgels_((const char*)'N', &n, &m, &nrhs, A, &lda, B, &ldb, workSpace, work, 
            &info);
    return info;
}

//extern void sgels_( char* trans, int* m, int* n, int* nrhs, float* a, int* lda,
//                float* b, int* ldb, float* work, int* lwork, int* info );


/**
 *@details BandPassRecorder 
 */
BandPassRecorder::BandPassRecorder( const ConfigNode& config )
    : AbstractModule( config )
{
    _requiredSamples = config.getOption("requiredSamples", "value", "20000").toULong();
    _polyDegree = config.getOption("fitParmaters", "coefficients", "3" ).toUInt();
    _totalSamples = 0;
}

/**
 *@details
 */
BandPassRecorder::~BandPassRecorder() {
}

void BandPassRecorder::_reset(const BinMap& map) {
    unsigned int size = map.numberBins();
    _sum.resize(size);
    _freq.resize(size);
    _freqMatrix.resize(size * _polyDegree);
    _valueMatrix.resize( _polyDegree );

    for(unsigned int i=0; i < size; ++i ) {
        int index=i*_polyDegree;
        _sum[i] = 0.0f;
        _freq[i] = map.binAssignmentNumber(i);
        for( int p=0; p < _polyDegree; ++p ) {
             _freqMatrix[index + p] = std::pow(_freq[i], p);
        }
    }
    _totalSamples = 0;
    _fit.resize(_polyDegree);
}

void BandPassRecorder::run( SpectrumDataSetStokes* stokesI, BandPass* bp ) {
        unsigned nSamples = stokesI->nTimeBlocks();
        unsigned nSubbands = stokesI->nSubbands();
        unsigned nChannels = stokesI->nChannels();
        unsigned nPolarisations = stokesI->nPolarisations();
        unsigned nBins = nChannels * nSubbands;

        float* I = stokesI->data();

        // TODO - initialise this object
        BinMap map(nBins);

        // if data type changes between calls then reset
        // all accumulative variables
        if( _sum.size() != nBins ) _reset(map);

        // calculate the integrated spectrum
        for (unsigned t = 0; t < nSamples; ++t) {
            for (unsigned s = 0; s < nSubbands; ++s) {
                int bin = (s * nChannels) - 1;
                long index = stokesI->index(s, nSubbands, 
                        0, nPolarisations,
                        t, nChannels ); 
                for (unsigned c = 0; c < nChannels; ++c) {
                    ++bin;
                    _sum[bin+c] += I[index + c];
                }
            }
        }
        _totalSamples += nSamples;
        if( _totalSamples < _requiredSamples ) return; // collect more data

        // get averaged values
        for (unsigned t = 0; t < nBins; ++t) {
            _sum[t] /= _totalSamples;
        }

        // start fitting to integrated spectrum
        int trimmed = 0;
        std::vector<float> residual(nBins);
        do {
            _polyFit(  &_sum[0], nBins );
            // trim away outliers and refit
            // points > 2sigma 
            float rsum = 0.0f;
            float rsumSquared = 0.0f;
            for (unsigned s = 0; s < nBins; ++s) {
                residual[s] = _sum[s] - _theFit(_freq[s]);
                rsum += residual[s];
                rsumSquared += std::pow(residual[s],2);
            }
            float resRms = std::sqrt(rsumSquared/nBins - std::pow((rsum/nBins),2));
            float margin = 2.0f * resRms;
            for (unsigned s = 0; s < nBins; ++s) {
                if( residual[s] > margin ) {
                    // chop this value out, setting it to the value of
                    // the fit function
                    ++trimmed;
                    _sum[s] = _theFit(_freq[s]) * _totalSamples;
                }
            }
        } while( trimmed ); // iterate until no outliers left
        bp->setData(map, QVector<float>::fromStdVector(_fit ));
        _reset(map);
}

float BandPassRecorder::_theFit(float v) const
{
   float tot = 0.0;
   for(unsigned int i=0; i< _fit.size(); ++i ) {
       tot += _fit[i]*std::pow(v,i);
   }
   return tot;
}

void BandPassRecorder::_polyFit( float* y, int nDataPoints )
{
    
    // make a copy of the input data
    for( int t=0; t < nDataPoints; ++t ) {
        _valueMatrix[t] = y[t];
    }
    int nrhs = 1, lda = nDataPoints, ldb = nDataPoints;
     
    // Workspace and status variables:
    float wkopt;
    float *work; //= workSize;
    int lwork = -1;
    int info;
    info = sgels( nDataPoints, _polyDegree, nrhs, 
            &_freqMatrix[0], lda, &_valueMatrix[0], ldb, 
            &wkopt, &lwork );
    lwork = (int)wkopt;
    work = (float*)malloc( lwork*sizeof(float) );

    /* Solve the equations A*X = B */
    info = sgels( nDataPoints, _polyDegree, nrhs, 
            &_freqMatrix[0], lda, &_valueMatrix[0], ldb, 
            work, &lwork );
    free(work);

    /* Check for the full rank */
    if( info > 0 ) {
            std::cerr << "The diagonal element %i of the triangular factor " 
                      << info;
            std::cerr << "of A is zero, so that A does not have full rank;\n"
                      << "the least squares solution could not be computed.\n";
            exit( 1 );
    } 
    // answer has overwritten _valueMatrix
    //std::vector<float>(_polyDegree) w; w.assign(_valueMatrix,_valueMatrix+_polyDegree);
    _fit.assign(&_valueMatrix[0],&_valueMatrix[0]+_polyDegree);
}

} // namespace lofar
} // namespace pelican
