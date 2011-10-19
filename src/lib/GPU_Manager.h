#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H


#include <QtCore/QThread>
#include <QtCore/QObject>
#include <QtCore/QList>
#include <QtCore/QSet>
#include "GPU_Resource.h"

/**
 * @file GPU_Manager.h
 */

namespace pelican {
namespace lofar {
class GPU_Resource;
class GPU_Job;

/**
 * @class GPU_Manager
 *  
 * @brief
 *    A GPU Reosurce Manager and Job Queue
 * @details
 * 
 */

class GPU_Manager : public QThread
{
    Q_OBJECT

    public:
        GPU_Manager( QObject* parent=0 );
        ~GPU_Manager();

        void submit( const GPU_Job& job ); 
        void addResource(GPU_Resource* r);

    protected:
        virtual void run();
        void _matchResources();

    protected slots:
        void _resourceFree();

    private:
        QList<GPU_Job> _queue;
        QList<GPU_Resource*> _resources;
        QList<GPU_Resource*> _freeResource;
};

} // namespace lofar
} // namespace pelican
#endif // GPU_MANAGER_H 
