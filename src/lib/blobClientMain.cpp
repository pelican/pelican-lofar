#include "PelicanBlobClient.h"

using namespace pelican;
using namespace pelican::lofar;

int main()
{
    PelicanBlobClient client("ChannelisedStreamData", "127.0.0.1", 6969);
    while (1)
        client.getData();
}
