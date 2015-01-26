#ifndef H5_LOFARBFVOLTAGEWRITER_H
#define H5_LOFARBFVOLTAGEWRITER_H

#ifdef HDF5_FOUND
#ifdef LOFAR_DAL_FOUND

#include "H5_LofarBFDataWriter.h"

/**
 * @file H5_LofarBFVoltageWriter.h
 */

namespace pelican {

namespace ampp {
class SpectrumDataSetC32;

/**
 * @class H5_LofarBFVoltageWriter
 *
 * @brief
 *    Class to write out h5 Lofar data format voltages
 * @details
 *
 */

class H5_LofarBFVoltageWriter : public H5_LofarBFDataWriter
{
    public:
        H5_LofarBFVoltageWriter( const ConfigNode& config );
        ~H5_LofarBFVoltageWriter();

    private:
        void _writeData(const SpectrumDataSetBase* data);

    private:
};

PELICAN_DECLARE(AbstractOutputStream, H5_LofarBFVoltageWriter)


} // namespace ampp
} // namespace pelican

#endif // LOFAR_DAL_FOUND
#endif // HDF5_FOUND
#endif // H5_LOFARBFVOLTAGEWRITER_H 
