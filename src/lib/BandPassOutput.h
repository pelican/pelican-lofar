#ifndef BANDPASSOUTPUT_H
#define BANDPASSOUTPUT_H


#include "pelican/output/AbstractOutputStream.h"
#include <QString>
#include <QList>

class QIODevice;

/**
 * @file BandPassOutput.h
 */

namespace pelican {
class ConfigNode;

namespace lofar {

/**
 * @class BandPassOutput
 *  
 * @configuration
 * <BandPassOutput>
 *  <file name="somefilename" />
 * </BandPassOutput>
 *
 * @brief
 *    creates a bandpass file
 * @details
 * 
 */

class BandPassOutput : public AbstractOutputStream
{
    public:
        BandPassOutput( const ConfigNode& configNode  );
        ~BandPassOutput();

        /// write out to a file with the specified name
        void addFile(const QString& filename);

    protected:
        virtual void sendStream(const QString& streamName, 
                                const DataBlob* dataBlob);
        QList<QIODevice*> _devices;

    private:
        QString _filename;
};

PELICAN_DECLARE(AbstractOutputStream, BandPassOutput )

} // namespace lofar
} // namespace pelican
#endif // BANDPASSOUTPUT_H 
