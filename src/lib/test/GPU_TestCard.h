#ifndef GPU_TESTCARD_H
#define GPU_TESTCARD_H


#include "GPU_Resource.h"
#include <QWaitCondition>
#include <QMutex>
#include <boost/bind.hpp>
#include <boost/function.hpp>

/**
 * @file GPU_TestCard.h
 */

namespace pelican {

namespace lofar {

/**
 * @class GPU_TestCard
 *  
 * @brief
 *    A dummy GPU_resource used for unit testing purposes
 * @details
 * 
 */

class GPU_TestCard : public GPU_Resource
{
    public:
        GPU_TestCard( );
        ~GPU_TestCard();

        // methods to query state
        //
        /// return the currently processeing job
        GPU_Job* currentJob() const;

        /// terminate the currently "processing" job
        void completeJob();

        /// termiate the current processing job by throwing
        template<typename T> void throwJob( const T& object ) {
            _doThrow = true;
            _object = boost::bind( &GPU_TestCard::doThrow<T>, this, object );
            completeJob();
        }

    protected: 
        template<typename T> void doThrow( const T object ) {
            throw object;
        }
        virtual void run( GPU_Job* job);

    private:
        GPU_Job* _current;
        QMutex _mutex;
        QWaitCondition _waitCondition;
        boost::function0<void> _object;
        bool _doThrow;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_TESTCARD_H 
