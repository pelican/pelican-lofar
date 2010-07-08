#include "viewer/DataBlobWidget.h"
#include "pelican/data/DataBlob.h"
#include <iostream>


namespace pelican {

namespace lofar {


/**
 *@details DataBlobWidget 
 */
DataBlobWidget::DataBlobWidget(QWidget* parent)
    : QWidget(parent)
{
}

/**
 *@details
 */
DataBlobWidget::~DataBlobWidget()
{
}

void DataBlobWidget::updateData( DataBlob* data)
{
    // do nothing here
    std::cout << "DataBlob received \n type :"
              << data->type().toStdString() 
              << " version:"
              << data->version().toStdString() 
              << std::endl;
}

} // namespace lofar
} // namespace pelican
