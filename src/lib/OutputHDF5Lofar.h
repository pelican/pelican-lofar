#ifndef HDF5OUTPUTLOFAR_H
#define HDF5OUTPUTLOFAR_H


#include "pelican/output/AbstractOutputStream.h"

/**
 * @file OutputHDF5Lofar.h
 */

namespace pelican {
    class ConfigNode;

namespace lofar {

/**
 * @class OutputHDF5Lofar
 *  
 * @brief
 *   Constructs Lofar hdf5 data files from Lofar Specific
 *   DataBlobs
 * @details
 * 
 */

class OutputHDF5Lofar : public AbstractOutputStream
{
    public:
        OutputHDF5Lofar(  const ConfigNode& configNode );
        ~OutputHDF5Lofar();
        void send(const QString& streamName, const DataBlob* dataBlob);

    private:
        QString _dir;
};

PELICAN_DECLARE(AbstractOutputStream, OutputHDF5Lofar )

} // namespace lofar
} // namespace pelican
#endif // HDF5OUTPUTLOFAR_H 
