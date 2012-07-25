#include "H5_LofarBFVoltageWriter.h"
#include "SpectrumDataSet.h"


namespace pelican {

namespace lofar {


/**
 *@details H5_LofarBFVoltageWriter 
 */
H5_LofarBFVoltageWriter::H5_LofarBFVoltageWriter( const ConfigNode& config )
    : H5_LofarBFDataWriter( config )
{
    _complexVoltages = true;
    if( _separateFiles ) {
        _stokesType = STOKES_I;
    } else {
        _stokesType = STOKES_XXYY;
    }
}

/**
 *@details
 */
H5_LofarBFVoltageWriter::~H5_LofarBFVoltageWriter()
{
}

void H5_LofarBFVoltageWriter::_writeData(const SpectrumDataSetBase* d ) {
    if( d->type() != "SpectrumDataSetC32" ) return;

    const SpectrumDataSetC32* spec = static_cast<const SpectrumDataSetC32*>(d);
    
    unsigned nSamples = spec->nTimeBlocks();
    unsigned nSubbands = spec->nSubbands();
    unsigned nChannels = spec->nChannels();
    unsigned nPolarisations = spec->nPolarisations();
    std::complex<float> const* data = spec->data();

    switch (_nBits) {
        case 32: {
             for (unsigned t = 0; t < nSamples; ++t) {
                 for (unsigned p = 0; p < nPolarisations; ++p ) {
                     for (int s = nSubbands - 1; s >= 0 ; --s) {
                         long index = spec->index(s, nSubbands, 
                                 p, nPolarisations, t, nChannels );
                         for(int i = nChannels - 1; i >= 0 ; --i) {
                             _file[p*2]->write(reinterpret_cast<const char*>(&data[index + i].real()), sizeof(float));
                             _file[p*2+1]->write(reinterpret_cast<const char*>(&data[index + i].imag()), sizeof(float));
                         }
                     }
                 }
             }
                 }
                 break;
        case 8: {
                for (unsigned t = 0; t < nSamples; ++t) {
                    for (unsigned p = 0; p < nPolarisations; ++p ) {
                        for (int s = nSubbands - 1; s >= 0 ; --s) {
                            long index = spec->index(s, nSubbands, 
                                    p, nPolarisations, t, nChannels );
                            for(int i = nChannels - 1; i >= 0 ; --i) {
                                int ci;
                                _float2int(&data[index + i].real(),&ci);
                                _file[p*2]->write((const char*)&ci,sizeof(unsigned char));
                                _float2int(&data[index + i].imag(),&ci);
                                _file[p*2+1]->write((const char*)&ci,sizeof(unsigned char));
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

} // namespace lofar
} // namespace pelican
