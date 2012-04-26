#include "LofarPelicanClientApp.h"
#include <boost/program_options.hpp>
#include <iostream>
#include <QtXml/QDomNode>
#include <QtXml/QDomNodeList>
#include <QSet>
#include "pelican/output/ThreadedDataBlobClient.h"


namespace opts = boost::program_options;
namespace pelican {

namespace lofar {


/**
 *@details LofarPelicanClientApp 
 */
LofarPelicanClientApp::LofarPelicanClientApp( int argc, char** argv, 
                                              const Config::TreeAddress& baseNode )
    : _address(baseNode)
{
    opts::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "provide this help message.")
        ("config,c", opts::value<std::string>(), "specify a configuration file");

    // Configuration option without a selection flag in the first argument
    // position is assumed to be a config file
    opts::positional_options_description p;
    p.add("config", -1);

    // Parse the command line arguments.
    opts::variables_map varMap;
    opts::store(opts::command_line_parser(argc, argv).options(desc)
            .positional(p).run(), varMap);
    opts::notify(varMap);

    // Check for help message.
    if (varMap.count("help")) {
        std::cout << desc << std::endl;
        exit(0);
    }

    // Get the configuration file name.
    std::string configFilename = "";
    if (varMap.count("config"))
        configFilename = varMap["config"].as<std::string>();
    if (!configFilename.empty())
        _config = pelican::Config(QString(configFilename.c_str()));

    // set up a client for each server
    Config::TreeAddress serversAddress = _address;
    serversAddress << Config::NodeId("servers","");
    ConfigNode servers = _config.get(serversAddress);
    QDomElement dom = servers.getDomElement();    
    QDomNodeList list = dom.elementsByTagName("server");
    for (int i = 0; i < list.size(); ++i) {
        QDomElement element = list.at(i).toElement();
        if( ! element.hasAttribute("name") ) {
            std::cout << "warning: unnamed server specified" << std::endl;
        }
        _clients.insert(element.attribute("name"), 
                        new ThreadedDataBlobClient(ConfigNode(element,0)));
    }

    // subscribe servers to the specified streams
    Config::TreeAddress streamsAddress = _address;
    streamsAddress << Config::NodeId("streams","");
    ConfigNode streams = _config.get(streamsAddress );
    QSet<QString> streamNames = streams.getOptionList("stream","name").toSet();
    foreach( ThreadedDataBlobClient* client, _clients ) {
        client->subscribe(streamNames);
    }
}

/**
 *@details
 */
LofarPelicanClientApp::~LofarPelicanClientApp()
{
    foreach( ThreadedDataBlobClient* client, _clients ) {
        delete client;
    }
}

QMap<QString,ThreadedDataBlobClient*> LofarPelicanClientApp::clients() const
{
    return _clients;
}

ConfigNode LofarPelicanClientApp::config(const Config::TreeAddress address) const
{
    return _config.get(address);
}

} // namespace lofar
} // namespace pelican
