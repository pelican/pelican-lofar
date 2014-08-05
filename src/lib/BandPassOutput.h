#ifndef BANDPASSOUTPUT_H
#define BANDPASSOUTPUT_H


#include "pelican/output/AbstractOutputStream.h"
#include <QtCore/QString>
#include <QtCore/QList>

class QIODevice;

/**
 * @file BandPassOutput.h
 */

namespace pelican {
class ConfigNode;

namespace ampp {

/**
 * @class BandPassOutput
 *
 * @brief
 *    Creates a bandpass file
 *
 * @details Configuration option :
 @verbatim
<BandPassOutput>
  <file name="somefilename" />
</BandPassOutput>
 @endverbatim
 *
 */

class BandPassOutput : public AbstractOutputStream
{
    public:
    /// Constructor
        BandPassOutput( const ConfigNode& configNode  );

    /// Destructor
        ~BandPassOutput();

        /// Write out to a file with the specified name
        void addFile(const QString& filename);

    protected:
        virtual void sendStream(const QString& streamName,
                                const DataBlob* dataBlob);
        QList<QIODevice*> _devices;

    private:
        QString _filename;
};

PELICAN_DECLARE(AbstractOutputStream, BandPassOutput )

} // namespace ampp
} // namespace pelican
#endif // BANDPASSOUTPUT_H
