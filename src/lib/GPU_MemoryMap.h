#ifndef GPU_MEMORYMAP_H
#define GPU_MEMORYMAP_H
#include <vector>
#include <QVector>


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

class GPU_MemoryMap
{
    public:
        GPU_MemoryMap( void* host_address = 0 , unsigned long bytes = 0 );
        template<typename T>
        GPU_MemoryMap( std::vector<T>& vec ) {
            _set(_host=&vec[0], vec.size() * sizeof(T) );
        }
        template<typename T>
        GPU_MemoryMap( QVector<T>& vec ) {
            _set(_host=&vec[0], vec.size() * sizeof(T) );
        }
        virtual ~GPU_MemoryMap();
        inline void* hostPtr() const { return _host; };
        inline unsigned long size() const { return _size; }
        bool operator==(const GPU_MemoryMap&) const;
        inline unsigned int qHash() const { return _hash; }
        //template<typename T> value() const { return static_cast<T>(*_host) };

    protected:
        void _set(void* host_address, unsigned long bytes);

    private:
        void* _host;
        unsigned long _size;
        unsigned int _hash;
};

unsigned int qHash(const GPU_MemoryMap& key);
} // namespace lofar
} // namespace pelican
#endif // GPU_MEMORYMAP_H 
