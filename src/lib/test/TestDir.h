#ifndef TESTDIR_H
#define TESTDIR_H
#include <QMutex>
#include <QList>
#include <QString>
#include <QDir>


/**
 * @file TestDir.h
 */

namespace pelican {
namespace lofar {
namespace test {

/**
 * @class TestDir
 *  
 * @brief
 *    create and manage a temporary test directory
 * @details
 * 
 */

class TestDir
{
    public:
        // @param cleanup if true will cause the directory
        //        to be deleted on destruction
        TestDir( bool cleanup = true );
        // @param cleanup if true will cause the directory
        //        to be deleted on destruction
        // @param label will be included in the directory name
        //        to aid human recognition
        TestDir( const QString& label, bool cleanup = true );
        ~TestDir();

        QString absolutePath() const;

    private:
        const QString& dirname() const;
        QString _clean( const QString& string );

    private:
        bool _cleanup;
        QString _label; // include label in the dirname
        mutable QMutex _mutex;
        mutable QString _dirname;
        mutable QDir _dir;
        static QList<QString> _reserved; // keep a track of reserved dirnames
        static long _id;  // unique ids across all TestFiles
};

} // namespace test
} // namespace lofar
} // namespace pelican
#endif // TESTDIR_H 
