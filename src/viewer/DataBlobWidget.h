#ifndef DATABLOBWIDGET_H
#define DATABLOBWIDGET_H

#include <QtGui/QWidget>

/**
 * @file DataBlobWidget.h
 */

namespace pelican {

class DataBlob;

namespace lofar {

/**
 * @class DataBlobWidget
 *
 * @brief
 *   Base class for DataBase Widget viewers
 * @details
 *
 */

class DataBlobWidget : public QWidget
{
    Q_OBJECT

    public:
        DataBlobWidget( QWidget* parent=0 );
        virtual ~DataBlobWidget();
        virtual void updateData( DataBlob* data) = 0;

    private:
};

} // namespace lofar
} // namespace pelican
#endif // DATABLOBWIDGET_H
