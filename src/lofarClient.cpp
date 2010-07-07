#include <QCoreApplication>

#include "pelican/core/PipelineApplication.h"
#include "lib/LofarStreamDataClient.h"

/*!
  \file lofarClient.cpp
  \ingroup pelican_lofar
*/
  
int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    pelican::PipelineApplication pApp(argc, argv);
    //pApp.registerPipeline(new TestPipeline;
    pApp.setDataClient("LofarStreamDataClient");
    pApp.start();
}
