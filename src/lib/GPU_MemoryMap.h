#ifndef GPU_MEMORYMAP_H
#define GPU_MEMORYMAP_H
#include <vector>
#include <QVector>
#include <boost/function.hpp>


/**
 * @file GPU_MemoryMap.h
 */

namespace pelican {

namespace lofar {

/**
 * @class GPU_MemoryMap
 *  
 * @brief
 *     A base class to transfer mark out data 
 *     to be transfered between different devices
 * @details
 * 
 */

//
// Use this class to represent data that has to be uploaded
// to the GPU and the data is liable to change.
//
class GPU_MemoryMap
{
    public:
        typedef boost::function0<void> CallBackT;

    public:
        GPU_MemoryMap( void* host_address = 0 , unsigned long bytes = 0 );
        template<typename T>
        GPU_MemoryMap( std::vector<T>& vec ) {
            _set(_host=&vec[0], vec.size() * sizeof(T) );
        }
        template<typename T>
        GPU_MemoryMap( QVector<T>& vec ) {
            _set(_host=vec.data(), vec.size() * sizeof(T) );
        }
        virtual ~GPU_MemoryMap();
        inline void* hostPtr() const { return _host; };
        inline unsigned long size() const { return _size; }
        bool operator==(const GPU_MemoryMap&) const;
        inline unsigned int qHash() const { return _hash; }
        template<typename T> T value() const { return *(static_cast<T*>(_host)); }
        /// add a function to be called immediately after a sync to the device 
        //  has completed.
        void addCallBack( const CallBackT& fn ) { 
                         _callbacks.append(fn); };
        const QList<CallBackT>& callBacks() const { return _callbacks; };
        // run all callBacks (and destroy callback list)
        void runCallBacks() const;

    protected:
        void _set(void* host_address, unsigned long bytes);

    private:
        mutable QList<CallBackT> _callbacks;
        void* _host;
        unsigned long _size;
        unsigned int _hash;
};

//
// Use this class to represent data that has to be uploaded
// once only. i.e. the data does not change once uploaded
//
class GPU_MemoryMapConst : public GPU_MemoryMap
{
    public:
        GPU_MemoryMapConst( void* host_address = 0 , unsigned long bytes = 0 ) 
                : GPU_MemoryMap(host_address, bytes) {}
        template<typename T>
        GPU_MemoryMapConst( T vec ) : GPU_MemoryMap( vec ) {}
        ~GPU_MemoryMapConst() {}
};

//
// Use this class to represent data that has to be downloaded
// No upload to the device is associated with this class
// 
class GPU_MemoryMapOutput : public GPU_MemoryMap
{
    public:
        GPU_MemoryMapOutput( void* host_address = 0 , unsigned long bytes = 0 ) 
                : GPU_MemoryMap(host_address, bytes) {}
        template<typename T>
        GPU_MemoryMapOutput( T vec ) : GPU_MemoryMap( vec ) {}
        ~GPU_MemoryMapOutput() {}
};

//
// Use this class to represent data that has to be uploaded
// to and dowloaded from the device
//
class GPU_MemoryMapInputOutput : public GPU_MemoryMapOutput
{
    public:
        GPU_MemoryMapInputOutput( void* host_address = 0 , unsigned long bytes = 0 ) 
                : GPU_MemoryMapOutput(host_address, bytes) {}
        template<typename T>
        GPU_MemoryMapInputOutput( T vec ) : GPU_MemoryMapOutput( vec ) {}
        ~GPU_MemoryMapInputOutput() {}
};

unsigned int qHash(const GPU_MemoryMap& key);
} // namespace lofar
} // namespace pelican
#endif // GPU_MEMORYMAP_H 
