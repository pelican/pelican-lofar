#ifndef H5_LOFARBFSTOKESWRITER_H
#define H5_LOFARBFSTOKESWRITER_H


#include "H5_LofarBFDataWriter.h"

/**
 * @file H5_LofarBFStokesWriter.h
 */

namespace pelican {

namespace lofar {

/**
 * @class H5_LofarBFStokesWriter
 *  
 * @brief
 *   Write our Stokes data blobs in Lofar H5 format
 * @details
 * 
 */

class H5_LofarBFStokesWriter : public H5_LofarBFDataWriter
{
    public:
        H5_LofarBFStokesWriter( const ConfigNode& config );
        ~H5_LofarBFStokesWriter();

    private:
        virtual void _writeData( const SpectrumDataSetBase* d );
};

PELICAN_DECLARE(AbstractOutputStream, H5_LofarBFStokesWriter)

} // namespace lofar
} // namespace pelican
#endif // H5_LOFARBFSTOKESWRITER_H 
