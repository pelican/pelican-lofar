/*
 * Lofar Data Viewer
 *
 * Data File Viewer for Lofar DataBlob Files
 * it in real time.
 *
 * Copyright OeRC 2010
 *
 */

#include "viewer/FileDataViewer.h"
#include "viewer/SpectrumDataSetWidget.h"

#include "pelican/utility/Config.h"

#include <QtGui/QApplication>
#include <boost/program_options.hpp>

#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

namespace opts = boost::program_options;
using namespace pelican;
using namespace pelican::lofar;
using std::cout;
using std::endl;


// Prototype for function to create a pelican configuration XML object.
Config createConfig(const opts::variables_map&);
std::vector<std::string> getFiles(const opts::variables_map&);
opts::variables_map process_options(int argc, char** argv);


int main(int argc, char** argv)
{
    QApplication app(argc, argv);

    opts::variables_map options = process_options(argc,argv);
    Config config = createConfig(options);
    Config::TreeAddress address;
    address << Config::NodeId("DataViewer", "Widgets");

    try {
        FileDataViewer* dv = new FileDataViewer( &config );
        dv->setDataBlobViewer("SpectrumDataSetStokes", "SpectrumDataSetWidget");
        dv->addFiles( QVector<std::string>::fromStdVector(getFiles( options )) );
        dv->show();
        cout << "entering exec()" << endl;
        return app.exec();
    }
    catch (const QString err)
    {
        cout << "ERROR: " << err.toStdString() << endl;
    }
}



/**
 * @details
 * Create a Pelican Configuration XML document for the lofar data viewer.
 */
opts::variables_map process_options(int argc, char** argv)
{
    // Check that argc and argv are nonzero
    if (argc == 0 || argv == NULL) throw QString("No command line.");

    // Declare the supported options.
    opts::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce help message.")
        ("file,f", opts::value< std::vector<std::string> >(), "Read in DataBlob from the specified file.")
        ("config,c", opts::value<std::string>(), "Set configuration file.");

    // Configuration option without a selection flag in the first argument
    // position is assumed to be a config file
    opts::positional_options_description p;
    p.add("file", -1);

    // Parse the command line arguments.
    opts::variables_map varMap;
    opts::store(opts::command_line_parser(argc, argv).options(desc)
            .positional(p).run(), varMap);
    opts::notify(varMap);

    // Check for help message.
    if (varMap.count("help")) {
        cout << desc << endl;;
        exit(0);
    }
    return varMap;
}

Config createConfig(const opts::variables_map& varMap)
{
    // Get the configuration file name.
    std::string configFilename = "";
    if (varMap.count("config"))
        configFilename = varMap["config"].as<std::string>();

    pelican::Config config;
    if (!configFilename.empty())
        config = pelican::Config(QString(configFilename.c_str()));

    return config;
}

std::vector<std::string> getFiles(const opts::variables_map& varMap) 
{
    std::vector<std::string> files;
    if (varMap.count("file")) {
        files = varMap["file"].as< std::vector<std::string> >();
    }
    return files;
}

