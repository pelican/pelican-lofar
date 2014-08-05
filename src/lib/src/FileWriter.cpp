#include "FileWriter.h"
#include <QtConcurrentRun>
#include <fcntl.h>   // open
#include <unistd.h>  // read, write, close
#include <cstdio>    // BUFSIZ
#include <iostream>
#include <QMutexLocker>


namespace pelican {

namespace ampp {


/**
 *@details FileWriter 
 */
FileWriter::FileWriter()
     : _fileHandle(0),_pos(0)
{
    int maxBuffers = 3;
    _bufSize = 32768;
    for( int i=0; i < maxBuffers; ++i ) {
        _buffersList.append( new char[_bufSize] );
    }
    _buffers.reset( &_buffersList );
    _currentBuffer = _buffers.next();
}

/**
 *@details
 */
FileWriter::~FileWriter()
{
   if( _fileHandle ) {
       _flush( _currentBuffer, _pos );
       ::close(_fileHandle);
   }
   foreach( char* b, _buffersList ) {
       delete[] b;
   }
}

void FileWriter::open( const char* filename ) {
    _fileHandle = ::open( filename, O_WRONLY | O_CREAT , 0644);
}

void FileWriter::close() {
   if( _fileHandle ) ::close(_fileHandle);
}

void FileWriter::write( const char* buffer, size_t length ) {
   if( (_bufSize - _pos) < (int)length ) {
       Q_ASSERT( (int)length < _bufSize );
       // launch a thread to handle device comms
       QtConcurrent::run(this, &FileWriter::_swap, _currentBuffer, _pos);
       _currentBuffer = _buffers.next();
       _pos = 0;
   }
   memcpy( &_currentBuffer[_pos], buffer, length );
   _pos += length;
}

void FileWriter::_swap( char* buffer, size_t size) {
   _flush(buffer, size);
   _buffers.unlock(buffer);
}

long FileWriter::tellp() {
   QMutexLocker lock( &_mutex);
   return lseek(_fileHandle, 0, SEEK_CUR);
}

void FileWriter::flush() {
   if( _fileHandle ) _flush( _currentBuffer, _pos );
   _pos=0;
}

void FileWriter::_flush( char* buffer, size_t size ) {
   // write buffer
   QMutexLocker lock( &_mutex);
   ::write( _fileHandle, buffer, size );
}

} // namespace ampp
} // namespace pelican
