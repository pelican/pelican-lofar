#ifndef GPU_TESTCARD_H
#define GPU_TESTCARD_H


#include "GPU_Resource.h"

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

    protected: 
        virtual void run( GPU_Job* job);

    private:
        GPU_Job* _current;
        QMutex _mutex;
        QWaitCondition _waitCondition;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_TESTCARD_H 
