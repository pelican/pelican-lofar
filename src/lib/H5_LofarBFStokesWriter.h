#ifndef H5_LOFARBFSTOKESWRITER_H
#define H5_LOFARBFSTOKESWRITER_H

#ifdef HDF5_FOUND

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
 *    <H5_LofarBFStokesWriter>
 *        <params nPolsToWrite="1" >
 *              the number of stokes parameters to write
 *              - each param will be written to a separate
 *              data file.
 *        </params>
 *    </H5_LofarBFStokesWriter>
 *
 *   other html tags from the base class should also be set
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

#endif // HDF5_FOUND
#endif // H5_LOFARBFSTOKESWRITER_H 
