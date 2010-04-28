#include "ChannelisedStreamData.h"


namespace pelican {
namespace lofar {

PELICAN_DECLARE_DATABLOB(ChannelisedStreamData)


/**
 * @details
 * @return
 */
QByteArray ChannelisedStreamData::serialise() const
{
    return QByteArray();
}


/**
 * @details
 * @param blob
 */
void ChannelisedStreamData::deserialise(const QByteArray& blob)
{

}


} // namespace lofar
} // namespace pelican

