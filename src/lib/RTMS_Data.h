#ifndef RTMS_DATA_H
#define RTMS_DATA_H


#include "pelican/data/DataBlob.h"

/**
 * @file RTMS_Data.h
 */

namespace pelican {

namespace ampp {

/**
 * @class RTMS_Data
 *  
 * @brief
 * 
 * @details
 * 
 */

class RTMS_Data : public DataBlob
{
    public:
        RTMS_Data();
        ~RTMS_Data();
        int startTime() const { return _t1; }
        int endTime() const { return _t2; }

    private:
        int _t1;
        int _t2;
};

} // namespace ampp
} // namespace pelican
#endif // RTMS_DATA_H 
