#ifndef FILEWRITER_H
#define FILEWRITER_H

#include "LockingPtrContainer.hpp"
#include <QMutex>

/**
 * @file FileWriter.h
 */

namespace pelican {

namespace ampp {

/**
 * @class FileWriter
 *  
 * @brief
 *    ofstream like interface to low level
 *    functions and asyncronous write.
 * @details
 *     avoid overhead of ofstream for fast binary data
 *     output
 */

class FileWriter
{
    public:
        FileWriter(  );
        ~FileWriter();
        void write();
        void flush();
        long tellp();
        void open( const char* filename );
        void write( const char* buffer, size_t length );
        void close();

    private:
        void _swap( char* buffer, size_t size);
        void _flush( char* buffer, size_t size );

    private:
        int _fileHandle;
        int _pos;
        int _bufSize;
        char* _currentBuffer;
        QList<char*> _buffersList;
        LockingPtrContainer<char> _buffers;
        QMutex _mutex;
};

} // namespace ampp
} // namespace pelican
#endif // FILEWRITER_H 
