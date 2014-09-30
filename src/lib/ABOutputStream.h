#ifndef ABOUTPUTSTREAM_H
#define ABOUTPUTSTREAM_H

/**
 * @file ABOutputStream.h
 */

#include <QtCore/QList>

class QIODevice;

#include "pelican/output/AbstractOutputStream.h"

namespace pelican {
namespace ampp {

/**
 * @ingroup c_examples
 *
 * @class ABOutputStream
 *
 * @brief
 * Writes a ABData data blob to a CSV file.
 *
 * @details
 * Specify one or more files in the configuration XML with
 * <ABOutputStream>
 *     <file name="file1.csv">
 *     <file name="duplicatefile.csv">
 * </ABOutputStream>
 *
 */
class ABOutputStream : public AbstractOutputStream
{
    public:
        // ABOutputStream constructor.
        ABOutputStream(const ConfigNode& configNode);

        // ABOutputStream destructor.
        ~ABOutputStream();

        // Add a file for output to be saved.
        void addFile(const QString& filename);

    protected:
        // Sends the data blob to the output stream.
        void sendStream(const QString& streamName, const DataBlob* dataBlob);

    private:
        QList<QIODevice*> _devices;
};

PELICAN_DECLARE(AbstractOutputStream, ABOutputStream)

} // namespace ampp
} // namespace pelican

#endif // ABOUTPUTSTREAM_H

