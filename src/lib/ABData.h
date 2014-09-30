#ifndef ABDATA_H
#define ABDATA_H

#include "pelican/data/DataBlob.h"
#include <vector>

namespace pelican {
namespace ampp {

/*
 * This data blob holds an array of floating-point time-stream data.
 */
class ABData : public DataBlob
{
    public:
        // Constructs a signal data blob.
        ABData() : DataBlob("ABData") {}

        // Returns a const pointer to the start of the data.
        const float* ptr() const {return (_data.size() > 0 ? &_data[0] : NULL);}

        // Returns a pointer to the start of the data.
        float* ptr() { return (_data.size() > 0 ? &_data[0] : NULL); }

        // Resizes the data store provided by the data blob.
        void resize(unsigned length) { _data.resize(length); }

        // Returns the size of the data.
        unsigned size() const { return _data.size(); }
    private:
        std::vector<float> _data; // The actual data array.
};

PELICAN_DECLARE_DATABLOB(ABData)

} // namespace ampp
} // namespace pelican

#endif // ABDATA_H

