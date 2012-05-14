#ifndef LOCKINGPTRCONTAINER_H
#define LOCKINGPTRCONTAINER_H
#include <QList>
#include <QWaitCondition>
#include <QMutex>
#include <QMutexLocker>


/**
 * @file LockingPtrContainer.h
 */

namespace pelican {

namespace lofar {

/**
 * @class LockingPtrContainer
 *  
 * @brief
 *    Template class to provide locks to a container of pointers to resources
 * @details
 * 
 */

template<typename T>
class LockingPtrContainer
{
    public:
        LockingPtrContainer() : _dataBuffer(0) {};
        LockingPtrContainer( QList<T*>* dataBuffer ) { reset(dataBuffer); };
        ~LockingPtrContainer() { _waitCondition.wakeAll(); };

        /// set the dataBuffer to manage
        void reset(QList<T*>* dataBuffer ) {
             _dataBuffer = dataBuffer;
             _available.clear();
             for(int i=0; i < dataBuffer->size(); ++i ) {
                _available.append( (*dataBuffer)[i] );
             }
        }

        /// return a reference to the next free resource
        //  This will block until a resource becomes available
        T* next() {
           QMutexLocker lock(&_mutex);
           while( ! _available.size() ) {
               _waitCondition.wait(&_mutex);
           }
           return const_cast<T*>(_available.takeFirst());
        }

        // unlock the specified data
        void unlock( const T* data ) {
           QMutexLocker lock(&_mutex);
           _available.append(data);
           _waitCondition.wakeOne();
        }

        QList<T*>* rawBuffer() const {
           return _dataBuffer;
        }

        int numberAvailable() const {
           QMutexLocker lock(&_mutex);
           return _available.size();
        }

        bool allAvailable() const {
           if( _dataBuffer )
               return _available.size() == _dataBuffer->size();
           return true;
        }

    private:
        QList<T*>* _dataBuffer;
        QWaitCondition _waitCondition;
        mutable QMutex _mutex;
        QList<const T*> _available; // array of available pointers
};

} // namespace lofar
} // namespace pelican
#endif // LOCKINGPTRCONTAINER_H 
