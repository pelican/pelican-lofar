#ifndef CHANNELISERPOLYPHASE_H
#define CHANNELISERPOLYPHASE_H

/**
 * @file ChanneliserPolyphase.h
 */

#include "pelican/modules/AbstractModule.h"

namespace pelican {

class ConfigNode;

namespace lofar {

class TimeStreamData;
class ChannelisedStreamData;

/**
 * @class ChanneliserPolyphase
 *
 * @brief
 * Module to channelise a time stream data blob.
 *
 * @details
 * Channelises time stream data using a polyphase channelising filter.
 */

class ChanneliserPolyphase : public AbstractModule
{
    public:
        /// Constructs the channeliser module.
        ChanneliserPolyphase(const ConfigNode& config);

        /// Destorys the channeliser module.
        ~ChanneliserPolyphase() {}

        /// Method converting the time stream to a spectrum.
        void run(const TimeStreamData* timeData, ChannelisedStreamData* spectrum);

    private:

};


}// namespace lofar
}// namespace pelican

#endif // TIMESTREAMDATA_H_
