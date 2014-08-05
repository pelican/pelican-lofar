#ifndef DEDISPERSIONEVENT_H
#define DEDISPERSIONEVENT_H


/**
 * @file DedispersionEvent.h
 */

namespace pelican {

namespace ampp {
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
  DedispersionEvent( int dmIndex, unsigned timeIndex, const DedispersionSpectra* data, float mfBinFactor, float mfBinValue );
        ~DedispersionEvent();
        unsigned timeBin() const;
        double getTime() const;
        float dm() const;
        float amplitude() const;
        float mfValue() const;
        float mfBinning() const;

    private:
        int _dm;
        unsigned _time;
        const DedispersionSpectra* _data;
        float _mfBinValue, _mfBinFactor;
};

} // namespace ampp
} // namespace pelican
#endif // DEDISPERSIONEVENT_H 
