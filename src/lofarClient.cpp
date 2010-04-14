#include <QCoreApplication>
#include "pelican/core/PipelineApplication.h"
#include "lib/LofarStreamDataClient.h"

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);
    PipelineApplication pApp(argc, argv);
    //pApp.registerPipeline(new TestPipeline;
    pApp.setDataClient("LofarStreamDataClient");
    pApp.start();
}
