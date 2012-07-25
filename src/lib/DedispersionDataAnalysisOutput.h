#ifndef DEDISPERSIONDATAANALYSISOUTPUT_H
#define DEDISPERSIONDATAANALYSISOUTPUT_H


#include "pelican/output/AbstractOutputStream.h"
#include <QTextStream>
#include <QString>

/**
 * @file DedispersionDataAnalysisOutput.h
 */

namespace pelican {
class ConfigNode;

namespace lofar {

/**
 * @class DedispersionDataAnalysisOutput
 *  
 * @brief
 *     Basic File Output for DedispersionDataAnalysis
 * @details
 * 
 */

class DedispersionDataAnalysisOutput : public AbstractOutputStream
{
    public:
        DedispersionDataAnalysisOutput( const ConfigNode& configNode );
        ~DedispersionDataAnalysisOutput();
        void addFile(const QString& filename);

    protected:
        virtual void sendStream(const QString& streamName,
                                const DataBlob* dataBlob);

    private:
        QList<QTextStream*> _streams;
        QList<QIODevice*> _devices;
        time_t _epoch;
        int _indexOfDump;
};

PELICAN_DECLARE(AbstractOutputStream, DedispersionDataAnalysisOutput)
} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONDATAANALYSISOUTPUT_H 
