#include "ChanneliserPolyphase.h"

#include "TimeStreamData.h"
#include "ChannelisedStreamData.h"

#include "pelican/utility/ConfigNode.h"

namespace pelican {
namespace lofar {

// Declare this class as a pelican module.
PELICAN_DECLARE_MODULE(ChanneliserPolyphase)


/**
 * @details
 * Constructor.
 *
 * @param[in] config XML configuration node.
 */
ChanneliserPolyphase::ChanneliserPolyphase(const ConfigNode& config)
: AbstractModule(config)
{
    // Get options from the config.

}

/**
 * @details
 * Method to run the channeliser.
 *
 * @param[in]  timeData Buffer of time samples to be channelised.
 * @param[out] spectrum Channels produced.
 */
void run(const TimeStreamData* timeData, ChannelisedStreamData* spectrum)
{
    // TODO: channelise the time stream...
}


}// namespace lofar
}// namespace pelican

