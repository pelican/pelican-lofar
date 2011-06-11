#include "BandPassRecorder.h"
#include <stdlib.h>
#include <clapack.h>

extern void sgels( char* trans, int* m, int* n, int* nrhs, float* a, int* lda,
                float* b, int* ldb, float* work, int* lwork, int* info );

namespace pelican {
namespace lofar {

/**
 *@details BandPassRecorder 
 */
BandPassRecorder::BandPassRecorder( ConfigNode& config )
    : AbstractModule( config )
{
    
}

/**
 *@details
 */
BandPassRecorder::~BandPassRecorder() {
}

void BandPassRecorder::run( SpectrumDataSetStokes* stokesI, BandPass* bp ) {
        unsigned nSamples = stokesI->nTimeBlocks();
        unsigned nSubbands = stokesI->nSubbands();
        unsigned nChannels = stokesI->nChannels();
        unsigned nPolarisations = stokesI->nPolarisations();
        unsigned nBins = nChannels * nSubbands;

        float* I = stokesI->data();

        // calculate the integrated spectrum
        for (unsigned t = 0; t < nSamples; ++t) {
            for (unsigned s = 0; s < nSubbands; ++s) {
                int bin = (s * nChannels) - 1;
                long index = stokesI->index(s, nSubbands, 
                        0, nPolarisations,
                        t, nChannels ); 
                for (unsigned c = 0; c < nChannels; ++c) {
                    ++bin;
                    sum[bin+c] += I[index + c];
                }
            }
        }
        _totalSamples += t;
        if( _totalSamples < requiredSamples ) return; // collect more data

        int trimmed = 0;
        do {
            // fit polynomial to integrated spectrum
            _polyfit(&freq, nFrequenceis, &sum, _polyDegree);
            // trim away outliers and refit
            // points > 2sigma 
            for (unsigned s = 0; s < nSubbands; ++s) {
                int bin = (s * nChannels) - 1;
                for (unsigned c = 0; c < nChannels; ++c) {
                    ++bin;
                    residual[bin+c] = sum[bin+c] - theFit[bin+c];
                    // margin = 2 sigma
                    if( residual[bin+c] > margin ) {
                        
                    }
                }
            }
        }
        while( trimmed )
        // take stats
        // itetrate until within tolerances
}

std::vector<float> BandPassRecorder::_polyfit(float* x, int nDataPoints, float* y, int deg)
{
     // create matrix for lapack lsf
     float* a = malloc( deg * nDataPoints * sizeof *X )
     float* rhs = malloc( nDataPoints * sizeof *rhs );
     for( int t=0; t < nDataPoints; ++t ) {
         for( int i=0; i < deg; ++i ) {
             X[index + i] = std::pow(x[index + i], i);
         }
     }
    int nrhs = 1, lda = nDataPoints, ldb = nDataPoints;
     
    // Workspace and status variables:
    float wkopt;
    float *work = workSize;
    int lwork = -1;
    int info = 0;
    sgels( "No transpose", &nDataPoints, &deg, &nrhs, a, &lda, rhs, &ldb, &wkopt, &lwork,
                    &info );
    lwork = (int)wkopt;
    work = (float*)malloc( lwork*sizeof(float) );
    /* Solve the equations A*X = B */
    sgels( "No transpose", &nDataPoints, &deg, &nrhs, a, &lda, rhs, &ldb, work, &lwork,
                    &info );
    /* Check for the full rank */
    if( info > 0 ) {
            printf( "The diagonal element %i of the triangular factor ", info );
            printf( "of A is zero, so that A does not have full rank;\n" );
            printf( "the least squares solution could not be computed.\n" );
            exit( 1 );
    } 
    // answer has overwritten rhs
    free(a);
    std::vector<float>(deg) w; w.assign(rhs,rhs+deg);
    free(rhs);
    return w;
}

} // namespace lofar
} // namespace pelican
