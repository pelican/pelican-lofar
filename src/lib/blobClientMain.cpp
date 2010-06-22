#include "PelicanBlobClient.h"

using namespace pelican;
using namespace pelican::lofar;

int main()
{
    PelicanBlobClient client("ChannelisedStreamData", "192.168.0.10", 6969);
    while (1)
        client.getData();
}
