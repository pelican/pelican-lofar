#ifndef LOCKINGCONTAINER_H
#define LOCKINGCONTAINER_H
#include <QList>
#include <QWaitCondition>
#include <QMutex>
#include <QMutexLocker>


/**
 * @file LockingContainer.h
 */

namespace pelican {

namespace lofar {

/**
 * @class LockingContainer
 *  
 * @brief
 *    Template class to provide locks to a container of resources
 * @details
 * 
 */

template<typename T>
class LockingContainer
{
    public:
        LockingContainer() : _size(0) {};
        LockingContainer( QList<T>* dataBuffer ) { reset(dataBuffer); };
        ~LockingContainer() {};

        /// set the dataBuffer to manage
        void reset(QList<T>* dataBuffer ) {
             _available.clear();
             _size = dataBuffer->size();
             for(int i=0; i < dataBuffer->size(); ++i ) {
                _available.append( &((*dataBuffer)[i]) );
             }
        }

        /// return a reference to the next free resource
        //  This will block until a resource becomes available
        T* next() {
           QMutexLocker lock(&_mutex);
           while( ! _available.size() ) {
               _waitCondition.wait(&_mutex);
           }
           return _available.takeFirst();
        }

        // unlock the specified data
        void unlock(T* data) {
           QMutexLocker lock(&_mutex);
           _available.append(data);
           _waitCondition.wakeOne();
        }

        // return true only if there are no locked objects
        bool allAvailable() const {
               return _available.size() == _size;
        }

    private:
        QWaitCondition _waitCondition;
        QMutex _mutex;
        int _size;
        QList<T*> _available; // array of available objects
};

} // namespace lofar
} // namespace pelican
#endif // LOCKINGCONTAINER_H 
