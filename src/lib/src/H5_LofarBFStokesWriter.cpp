#include "H5_LofarBFStokesWriter.h"
#include "SpectrumDataSet.h"


namespace pelican {

namespace ampp {


/**
 *@details H5_LofarBFStokesWriter 
 */
H5_LofarBFStokesWriter::H5_LofarBFStokesWriter( const ConfigNode& config )
    : H5_LofarBFDataWriter( config )
{
    _complexVoltages=false;
    if( _separateFiles ) {
        _stokesType = STOKES_I;
    } else {
        _stokesType = STOKES_IQUV;
    }

    // Number of polarisations components to write out, 1 - 4
    _setPolsToWrite(config.getOption("params", "nPolsToWrite", "1").toUInt());
}

/**
 *@details
 */
H5_LofarBFStokesWriter::~H5_LofarBFStokesWriter()
{
}

void H5_LofarBFStokesWriter::_writeData( const SpectrumDataSetBase* d )
{
    if( d->type() != "SpectrumDataSetStokes" ) return;

    const SpectrumDataSetStokes* stokes = static_cast<const SpectrumDataSetStokes*>(d);
    unsigned nSamples = stokes->nTimeBlocks();
    unsigned nSubbands = stokes->nSubbands();
    unsigned nChannels = stokes->nChannels();
    unsigned nPolarisations = stokes->nPolarisations();
    float const * data = stokes->data();

    switch (_nBits) {
        case 32: {
             for (unsigned t = 0; t < nSamples; ++t) {
                 for (unsigned p = 0; p < polsToWrite(); ++p) {
                     for (int s = nSubbands - 1; s >= 0 ; --s) {
                         long index = stokes->index(s, nSubbands, 
                                 p, nPolarisations, t, nChannels );
                         for(int i = nChannels - 1; i >= 0 ; --i) {
                             _file[p]->write(reinterpret_cast<const char*>(&data[index + i]), 
                                     sizeof(float));
                         }
                     }
                 }
             }
                 }
                 break;
        case 8: {
                for (unsigned t = 0; t < nSamples; ++t) {
                    for (unsigned p = 0; p < polsToWrite(); ++p) {
                        for (int s = nSubbands - 1; s >= 0 ; --s) {
                            long index = stokes->index(s, nSubbands, 
                                    p, nPolarisations, t, nChannels );
                            for(int i = nChannels - 1; i >= 0 ; --i) {
                                int ci;
                                _float2int(&data[index + i],&ci);
                                _file[p]->write((const char*)&ci,sizeof(unsigned char));
                            }
                        }
                    }
                }
            }
            break;
        default:
            throw(QString("H5_LofarStokesWriter: %1 bit datafiles not yet supported"));
            break;
    }
}


} // namespace ampp
} // namespace pelican
