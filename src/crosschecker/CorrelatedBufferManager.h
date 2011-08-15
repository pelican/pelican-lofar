#ifndef CORRELATEDBUFFERMANAGER_H
#define CORRELATEDBUFFERMANAGER_H
#include <QObject>
#include <QMap>
#include <QString>
#include "../lib/RTMS_Data.h"

/**
 * @file CorrelatedBufferManager.h
 */

namespace pelican {

namespace lofar {
class CorrelatingBuffer;

/**
 * @class CorrelatedBufferManager
 *  
 * @brief
 *    Manages a collection of correlating buffers
 *    
 * @details
 * 
 */

class CorrelatedBufferManager : public QObject
{ 
    Q_OBJECT

    public:
        CorrelatedBufferManager( QObject* parent = 0 );
        ~CorrelatedBufferManager();
        CorrelatingBuffer* newBuffer(const QString& buffer);
        void setCorrelationMargin(int delta) { _delta=delta; };


    signals:
        void foundCorrelation(QMap<QString, RTMS_Data>);

    private slots:
        void dataReceived(const RTMS_Data&);

    private:
        QMap<QString, RTMS_Data> _findCorrelated(const RTMS_Data& d, const CorrelatingBuffer* exclude);
    private:
        QMap<QString,CorrelatingBuffer* > _buffers;
        int _delta;
};

} // namespace lofar
} // namespace pelican
#endif // CORRELATEDBUFFERMANAGER_H 
