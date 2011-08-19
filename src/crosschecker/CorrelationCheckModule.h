#ifndef CORRELATIONCHECKMODULE_H
#define CORRELATIONCHECKMODULE_H


#include <QObject>
#include "pelican/core/AbstractModule.h"

/**
 * @file CorrelationCheckModule.h
 */

namespace pelican {

namespace lofar {
class RTMS_Data;

/**
 * @class CorrelationCheckModule
 *  
 * @brief
 *    Perform verification of the correlated data
 * @details
 * 
 */

class CorrelationCheckModule : public QObject, public AbstractModule
{
    Q_OBJECT

    public:
        CorrelationCheckModule( const ConfigNode config );
        ~CorrelationCheckModule();

    public slots:
        void run(const QMap<QString, RTMS_Data>& map);

    private:
};

} // namespace lofar
} // namespace pelican
#endif // CORRELATIONCHECKMODULE_H 
