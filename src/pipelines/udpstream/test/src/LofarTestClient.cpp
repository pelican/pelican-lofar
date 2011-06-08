#include "LofarTestClient.h"
#include "EmulatorPipeline.h"


namespace pelican {

namespace lofar {


/**
 *@details LofarTestClient 
 */
LofarTestClient::LofarTestClient( LofarEmulatorDataSim* e, const QFile& configFile,
                                  const QString& stream, QObject* parent)
    : QThread(parent), _emulator(e), _pipeline(0), _stream(stream)
{
    _configFile.setFileName( configFile.fileName());
    if( ! _configFile.exists() ) {
        throw QString("LofarTestClient: Configfile does not exist");
    }
}

/**
 *@details
 */
LofarTestClient::~LofarTestClient()
{
    terminate();
    wait();
}

void LofarTestClient::startup()
{
    start();
    while( ! _pipeline || ! _pApp->isRunning() ) { msleep(5); }
}

void LofarTestClient::run()
{
    int argc=2;
    char* argv[argc];
    argv[0] = (char*)"LofarTestClient";
    std::string file = _configFile.fileName().toStdString();
    argv[1] = new char[file.size() + 1];
    strcpy(argv[1], file.c_str());

     // start up the pipeline application
    try {
        // Create a PipelineApplication.
        _pApp.reset(new PipelineApplication(argc, argv));

        // Set the data client.
        _pApp->setDataClient("PelicanServerClient");

        // Register the pipelines that can run.
        _pipeline = new EmulatorPipeline( _stream, _emulator );
        _pApp->registerPipeline(_pipeline);

        // Start the pipeline driver.
        _pApp->start();
    }
    catch (QString err) {
        _pipeline = 0;
        delete[] argv[1];
        std::cerr << "error in LofarTestClient:: " << err.toStdString() << std::endl;
        throw( QString("Error caught in LofarTestClient.cpp: ") + err );
        //exit(1);
    }
    delete[] argv[1];
    _pipeline = 0;
}

unsigned long LofarTestClient::count() const
{
     if( _pipeline ) 
         return _pipeline->count();
     return 0;
}

} // namespace lofar
} // namespace pelican
