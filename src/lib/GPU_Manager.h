#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H


#include <QtCore/QThread>
#include <QtCore/QList>
#include "GPU_Job.h"

/**
 * @file GPU_Manager.h
 */

namespace pelican {
namespace lofar {

/**
 * @class GPU_Manager
 *  
 * @brief
 * 
 * @details
 * 
 */

class GPU_Manager : public QThread
{
    Q_OBJECT

    public:
        GPU_Manager( QObject parent=0 );
        ~GPU_Manager();

        void submit( const GPU_Job& job) { _queue.append(job); }; 

    protected:
        virtual void exec();
        void runJob(const GPU_Job&);

    private:
        QList<GPU_Job> _queue;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_MANAGER_H 
