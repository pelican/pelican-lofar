#include "H5_LofarBFVoltageWriter.h"
#include "SpectrumDataSet.h"


namespace pelican {

namespace ampp {


/**
 *@details H5_LofarBFVoltageWriter 
 */
H5_LofarBFVoltageWriter::H5_LofarBFVoltageWriter( const ConfigNode& config )
    : H5_LofarBFDataWriter( config )
{
    _complexVoltages = true;
    /*
    if( _separateFiles ) {
        _stokesType = STOKES_I;
    } else {
        _stokesType = STOKES_XXYY;
    }
    */
    _stokesType = STOKES_XXYY; // always the case for complex volts (Chris, we need to talk about this)
    
    _setPolsToWrite(4); // only support writing all 4 pols
                        // as _writeData() assumes this to work
}

/**
 *@details
 */
H5_LofarBFVoltageWriter::~H5_LofarBFVoltageWriter()
{
}

void H5_LofarBFVoltageWriter::_writeData(const SpectrumDataSetBase* d ) {
    typedef std::complex<float> Complex; 
    if( d->type() != "SpectrumDataSetC32" ) return;

    const SpectrumDataSetC32* spec = static_cast<const SpectrumDataSetC32*>(d);
    
    unsigned nSamples = spec->nTimeBlocks();
    unsigned nSubbands = spec->nSubbands();
    unsigned nChannels = spec->nChannels();
    unsigned nPolarisations = spec->nPolarisations();
    //    float const* data = (const float*)spec->data();
    const Complex* data = (const Complex*)spec->data();
    const Complex* dataPol;

    switch (_nBits) {
        case 32: {
             for (unsigned t = 0; t < nSamples; ++t) {
                 for (unsigned p = 0; p < nPolarisations; ++p ) {
                     int pindex=p*2;
                     for (int s = 0; s < nSubbands; ++s) {
                         long index = spec->index(s, nSubbands, 
                                                  p, nPolarisations, t, nChannels );// * 2;
                         dataPol = &data[index];
                         for(int i = 0; i < nChannels ; ++i) {
                           /*
                             _file[pindex]->write(reinterpret_cast<const char*>(&data[index + i]), sizeof(float));
                             _file[pindex+1]->write(reinterpret_cast<const char*>(&data[index + i + 1]), sizeof(float));
                           */                             

                           _file[pindex]->write(reinterpret_cast<const char*>(&dataPol[i].real()), sizeof(float));
                           _file[pindex+1]->write(reinterpret_cast<const char*>(&dataPol[i].imag()), sizeof(float));
                         }
                     }
                 }
             }
                 }
                 break;
                 /*
        case 8: {
                for (unsigned t = 0; t < nSamples; ++t) {
                    for (unsigned p = 0; p < nPolarisations; ++p ) {
                        for (int s = nSubbands - 1; s >= 0 ; --s) {
                            long index = spec->index(s, nSubbands, 
                                    p, nPolarisations, t, nChannels ) * 2;
                            for(int i = nChannels - 1; i >= 0 ; --i) {
                                int ci;
                                _float2int(&data[index + i],&ci);
                                _file[p*2]->write((const char*)&ci,sizeof(unsigned char));
                                _float2int(&data[index + i + 1],&ci);
                                _file[p*2+1]->write((const char*)&ci,sizeof(unsigned char));
                            }
                        }
                    }
                }
            }
            break;
                 */
        default:
            throw(QString("H5_LofarStokesWriter: %1 bit datafiles not yet supported"));
            break;
    }
}

} // namespace ampp
} // namespace pelican
