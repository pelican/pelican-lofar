#ifndef SigprocAdapter_H
#define SigprocAdapter_H

#include "pelican/core/AbstractStreamAdapter.h"
#include "SpectrumDataSet.h"
#include "file_handler.h"
#include <complex>

using namespace pelican;
using namespace pelican::lofar;

class SigprocAdapter: public AbstractStreamAdapter
{
    public:
        /// Constructs a new SigprocAdapter.
        SigprocAdapter(const ConfigNode& config);

        /// Destroys the SigprocAdapter.
        ~SigprocAdapter() {}

    protected:
        /// Method to deserialise a sigproc file
        void deserialise(QIODevice* in);

    private:
        /// Updates and checks the size of the time stream data.
        void _checkData();

    private:
        SpectrumDataSetStokes* _stokesData;
        FILE_HEADER* _header;
        FILE *_fp;

        unsigned _nSamples;
        unsigned _nSubbands;
        unsigned _nPolarisations;
        unsigned _nBits;
        double   _tsamp;
        unsigned long int _iteration;
};

PELICAN_DECLARE_ADAPTER(SigprocAdapter)

#endif // SigprocAdapter_H
