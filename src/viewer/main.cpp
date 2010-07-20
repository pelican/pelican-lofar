/*
 * Lofar Data Viewer
 *
 * Attaches to a data stream from the Lofar pipeline to display
 * it in real time.
 *
 * Copyright OeRC 2010
 *
 */

#include <iostream>
#include <cstdlib>
#include <QtGui/QApplication>
#include <boost/program_options.hpp>
#include "pelican/utility/Config.h"
#include "viewer/LofarDataViewer.h"
#include <iostream>

namespace opts = boost::program_options;

pelican::Config createConfig(int argc, char** argv)
{
    // Check that argc and argv are nonzero
    if (argc == 0 || argv == NULL) {
        throw QString("No command line.");
    }

    // Declare the supported options.
    opts::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce help message.")
        ("config,c", opts::value<std::string>(), "Set configuration file.")
        ("port,p", opts::value<unsigned>(), "port.")
        ("address,a", opts::value<std::string>(), "port.");


    // Configuration option without a selection flag in the first argument
    // position is assumed to be a fits file
    opts::positional_options_description p;
    p.add("config", -1);

    // Parse the command line arguments.
    opts::variables_map varMap;
    opts::store(opts::command_line_parser(argc, argv).options(desc).positional(p).run(), varMap);
    opts::notify(varMap);

    // Check for help message.
    if (varMap.count("help")) {
        std::cout << desc << "\n";
        exit(0);
    }

    // Get the configuration file name.
    std::string configFilename = "";
    if (varMap.count("config")) {
        configFilename = varMap["config"].as<std::string>();
        //std::cout << "--- " << configFilename << std::endl;
    }

    pelican::Config config;
    if (!configFilename.empty()) {
        config = pelican::Config(QString(configFilename.c_str()));
    }

    pelican::Config::TreeAddress baseAddress;
    baseAddress << pelican::Config::NodeId("DataViewer", "");

    unsigned port = 0;
    if (varMap.count("port")) {
        port = varMap["port"].as<unsigned>();
        pelican::Config::TreeAddress a;
        a << pelican::Config::NodeId("DataViewer", "");
        a << pelican::Config::NodeId("server", "");
        config.setAttribute(a, "port", QString::number(port));

    }

    std::string address = "";
    if (varMap.count("address")) {
        address = varMap["address"].as<std::string>();
        pelican::Config::TreeAddress a;
        a << pelican::Config::NodeId("DataViewer", "");
        a << pelican::Config::NodeId("server", "");
        config.setAttribute(a, "address", QString::fromStdString(address));
    }

    return config;
}



int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    try {
        pelican::Config config = createConfig(argc, argv);

        pelican::Config::TreeAddress address;
        address << pelican::Config::NodeId("DataViewer", "");

//        config.save("config.xml");
//        config.summary();

        pelican::lofar::LofarDataViewer ldv(config.get(address));
        ldv.show();
    }
    catch (QString err) {
        std::cout << "ERROR: " << err.toStdString() << std::endl;
    }

    return app.exec();
}

