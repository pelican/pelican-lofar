#include "pelican/core/PipelineApplication.h"
#include "DedispersionApplication.h"
#include "PumaOutput.h"
#include <iostream>

using std::cout;
using std::endl;
using namespace pelican;
using namespace pelican::ampp;

int main(int argc, char* argv[])
{
    QString stream = "LofarTimeStream2";

    try {
        DedispersionApplication app(argc, argv,stream);
    }
    catch (const QString& err) {
        std::cout << "Error caught in dedispersionStream1.cpp: " << err.toStdString() << endl;
    }

    return 0;
}
