#ifndef JTESTOUTPUTSTREAM_H
#define JTESTOUTPUTSTREAM_H

/**
 * @file JTestOutputStream.h
 */

#include <QtCore/QList>

class QIODevice;

#include "pelican/output/AbstractOutputStream.h"

namespace pelican {
namespace ampp {

/**
 * @ingroup c_examples
 *
 * @class JTestOutputStream
 *
 * @brief
 * Writes a JTestData data blob to a CSV file.
 *
 * @details
 * Specify one or more files in the configuration XML with
 * <JTestOutputStream>
 *     <file name="file1.csv">
 *     <file name="duplicatefile.csv">
 * </JTestOutputStream>
 *
 */
class JTestOutputStream : public AbstractOutputStream
{
    public:
        // JTestOutputStream constructor.
        JTestOutputStream(const ConfigNode& configNode);

        // JTestOutputStream destructor.
        ~JTestOutputStream();

        // Add a file for output to be saved.
        void addFile(const QString& filename);

    protected:
        // Sends the data blob to the output stream.
        void sendStream(const QString& streamName, const DataBlob* dataBlob);

    private:
        QList<QIODevice*> _devices;
};

PELICAN_DECLARE(AbstractOutputStream, JTestOutputStream)

} // namespace ampp
} // namespace pelican

#endif // JTESTOUTPUTSTREAM_H

