#ifndef CORRELATINGBUFFER_H
#define CORRELATINGBUFFER_H

#include <QtCore/QObject>
#include <QtCore/QString>
#include <QtCore/QMap>
#include <QtCore/QHash>
#include "../lib/RTMS_Data.h"

/**
 * @file CorrelatingBuffer.h
 */

namespace pelican {
class Stream;

namespace lofar {
class CorrelatedBufferManager;

/**
 * @class CorrelatingBuffer
 *
 * @brief
 *     Takes RTMS datablobs and correlates against the timestamps with
 *     other RTMS datablobs
 * @details
 *
 */

class RTMS_Data;

class CorrelatingBuffer : public QObject
{
    Q_OBJECT

    private:
        CorrelatingBuffer( CorrelatedBufferManager* _manager, 
                           QObject* parent = 0 );

    public:
        ~CorrelatingBuffer();
        void add(RTMS_Data&);

    protected:
        void doSomething(const QMap<QString, RTMS_Data>) const;

    signals:
        void dataAdded(const RTMS_Data&);

    protected slots:
        void newData(const Stream& stream);

    private:
        typedef long Timestamp_T;
        QString _name;
        QHash<Timestamp_T,RTMS_Data> _buffer;
        CorrelatedBufferManager* _manager;

    friend class CorrelatedBufferManager;
};

} // namespace lofar
} // namespace pelican
#endif // CORRELATINGBUFFER_H
