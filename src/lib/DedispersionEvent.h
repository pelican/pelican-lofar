#ifndef DEDISPERSIONEVENT_H
#define DEDISPERSIONEVENT_H


/**
 * @file DedispersionEvent.h
 */

namespace pelican {

namespace lofar {
class DedispersionSpectra;

/**
 * @class DedispersionEvent
 *  
 * @brief
 *    Container class to provide API for a specific dedispersion event
 * @details
 * 
 */

class DedispersionEvent
{
    public:
        DedispersionEvent( int dmIndex, unsigned timeIndex, const DedispersionSpectra* data );
        ~DedispersionEvent();
        unsigned timeBin() const;
        float dm() const;
        float amplitude() const;

    private:
        int _dm;
        unsigned _time;
        const DedispersionSpectra* _data;
};

} // namespace lofar
} // namespace pelican
#endif // DEDISPERSIONEVENT_H 
