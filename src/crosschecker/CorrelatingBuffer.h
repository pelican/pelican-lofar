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

namespace lofar {

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

    public:
        CorrelatingBuffer( const QString& name, QMap<QString, CorrelatingBuffer*> buffers, QObject* parent = 0 );
        ~CorrelatingBuffer();

        void add(RTMS_Data&);
        QMap<QString, RTMS_Data> findCorrelated(const RTMS_Data&) const;

    protected:
        void doSomething(const QMap<QString, RTMS_Data>) const;

    signals:
        void foundCorrelation(QMap<QString, RTMS_Data>);

    private:
        typedef long Timestamp_T;
        QString _name;
        QHash<Timestamp_T,RTMS_Data> _buffer;
        int _delta;
        QMap<QString, CorrelatingBuffer*> _correlatedBuffers;
};

} // namespace lofar
} // namespace pelican
#endif // CORRELATINGBUFFER_H
