#include "TestDir.h"
#include <QDir>
#include <iostream>
#include <QMutexLocker>
#include <QCoreApplication>
#include <QFileInfo>


namespace pelican {

namespace lofar {
namespace test {

long TestDir::_id = 0;
QList<QString> TestDir::_reserved = QList<QString>();


/**
 *@details TestDir 
 */
TestDir::TestDir( bool cleanup )
    : _cleanup( cleanup )
{
    dirname();
}

TestDir::TestDir( const QString& label, bool cleanup )
    : _cleanup( cleanup )
{
    _label = _clean(label) + "_";
    dirname();
}

QString TestDir::_clean( const QString& dirty ) {
    QString string = dirty;
    string.replace(QChar('.'),QChar('_'));
    string.replace(QChar('/'),QChar('_'));
    return string;
}

/**
 *@details
 */
TestDir::~TestDir()
{
    if( _dir.exists() ) {
        if( _cleanup ) {
          foreach(QFileInfo info, _dir.entryInfoList(QDir::NoDotAndDotDot | QDir::Files )) {
          //std::cout << "remove file " << info.absoluteFilePath().toStdString();
             QFile::remove(info.absoluteFilePath());
          }
          //std::cout << "remove dir " << _dirname.toStdString();
          _dir.rmdir( _dirname );
        }
        else {
          std::cout << "TestDir: not removing temporary dir: "
                    << _dirname.toStdString() << std::endl;                      
        }
    }
    else {
        std::cout << "TestDir: no dir to clean:" << _dir.absolutePath().toStdString() << std::endl;
    }
}

const QString& TestDir::dirname() const {
    QFileInfo dirInfo;
    if( _dirname == "" ) {
        QMutexLocker lock( &_mutex );
        do {
            _dirname = QDir::tempPath() + "/TestDir_" + _label 
                + QString().setNum(QCoreApplication::applicationPid()) + "_" 
                + QString().setNum(++_id) ;                 
            dirInfo.setFile( _dirname );
        }
        while( dirInfo.exists() && _reserved.contains(_dirname) );
        _reserved.append(_dirname);
        if( ! QDir::temp().mkpath( _dirname ) )
            throw QString("TestDir: cannor create directory %1").arg(_dirname);
        _dir.setPath( _dirname );
    }
    return _dirname;
}

QString TestDir::absolutePath() const {
    return dirname();
}

} // namespace test
} // namespace lofar
} // namespace pelican
