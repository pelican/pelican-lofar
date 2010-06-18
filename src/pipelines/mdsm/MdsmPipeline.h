#ifndef MDSMPIPELINE_H
#define MDSMPIPELINE_H


#include "pelican/core/AbstractPipeline.h"
#include "pelican/data/DataBlob.h"
#include "ChanneliserPolyphase.h"
#include "PolyphaseCoefficients.h"
#include "ChannelisedStreamData.h"

/**
 * @file MdsmPipeline.h
 */

namespace pelican {
namespace lofar {

/**
 * @class MdsmPipeline
 *  
 * @brief
 * 
 * @details
 * 
 */
class MdsmPipeline : public AbstractPipeline
{
    public:
        MdsmPipeline(  );
        ~MdsmPipeline();

        /// Initialises the pipeline.
        void init();

        /// Runs the pipeline.
        void run(QHash<QString, DataBlob*>& remoteData);

    private:
        /// Module pointers
        ChanneliserPolyphase* channeliser;

        /// Local data blob
        PolyphaseCoefficients* polyphaseCoeff;
        ChannelisedStreamData* channelisedData;
};

} // namespace lofar
} // namespace pelican

#endif // MDSMPIPELINE_H 
