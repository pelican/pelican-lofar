#include <QtCore/QCoreApplication>

#include "pelican/emulator/EmulatorDriver.h"
#include "LofarEmulatorDataSim.h"
#include "pelican/utility/Config.h"
#include <boost/program_options.hpp>
#include <iostream>

namespace opts = boost::program_options;

using namespace pelican;
using namespace pelican::lofar;

pelican::Config createConfig(int argc, char** argv);

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);

    /*
    //unsigned interval = 500000; // microseconds
    unsigned interval = 2000; // microseconds
    unsigned startDelay = 1; // seconds

    unsigned nPols = 2;
    unsigned nSubbands = 61;

    QString xml =
            "<LofarEmulatorDataSim>"
            ""
            "    <connection host=\"127.0.0.1\" port=\"8090\"/>"
            ""
            "    <packetSendInterval value=\"%1\"/>"
            "    <packetStartDelay   value=\"%2\"/>"
            ""
            "    <subbandsPerPacket  value=\"%3\"/>"
            "    <polsPerPacket      value=\"%4\"/>"
            ""
            "</LofarEmulatorDataSim>";

    xml = xml.arg(interval);
    xml = xml.arg(startDelay);
    xml = xml.arg(nSubbands);
    xml = xml.arg(nPols);
    */
    try{
    pelican::Config config = createConfig(argc, argv);
    Config::TreeAddress emulatorAddress;
    emulatorAddress << Config::NodeId("emulators","") << Config::NodeId("LofarEmulatorDataSim","");
    ConfigNode configNode = config.get(emulatorAddress);


    //    ConfigNode config(xml);

    EmulatorDriver emulator(new LofarEmulatorDataSim(configNode));

    return app.exec();
    }
    catch( QString e) {
      std::cerr << "Emulator exception: " <<  e.toStdString() << std::endl; 
    }
}

/**                                                                                                                                      
 * @details                                                                                                                              
 * Create a Pelican Configuration XML document for the lofar data viewer.                                                                
 */
pelican::Config createConfig(int argc, char** argv)
{
  // Check that argc and argv are nonzero                                                                                              
  if (argc == 0 || argv == NULL) throw QString("No command line.");

  // Declare the supported options.                                                                                                    
  opts::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "Produce help message.")
    ("config,c", opts::value<std::string>(), "Set configuration file.");


  // Configuration option without a selection flag in the first argument                                                               
  // position is assumed to be a config file                                                                                           
  opts::positional_options_description p;
  p.add("config", -1);

  // Parse the command line arguments.                                                                                                 
  opts::variables_map varMap;
  opts::store(opts::command_line_parser(argc, argv).options(desc)
	      .positional(p).run(), varMap);
  opts::notify(varMap);

  // Check for help message
  if (varMap.count("help")) {
    std::cout << desc << std::endl;;
    exit(0);
  }

  // Get the configuration file name
  std::string configFilename = "";
  if (varMap.count("config"))
    configFilename = varMap["config"].as<std::string>();

  pelican::Config config;
  if (!configFilename.empty())
    config = pelican::Config(QString(configFilename.c_str()));

  return config;
}
