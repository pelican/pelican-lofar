#ifndef BINMAP_H
#define BINMAP_H


/**
 * @file BinMap.h
 */

namespace pelican {

namespace lofar {

/**
 * @class BinMap
 *  
 * @brief
 *    A description of the bins range and width for binned data
 * @details
 * 
 */

class BinMap
{
    public:
        BinMap();
        BinMap( unsigned int numberBins );
        ~BinMap();
        void setStart(float);
        void setBinWidth(float);
        int binIndex(float) const;
        float width() const { return _width; };
        unsigned int numberBins() const { return _nBins; };
        float startValue() const { return _lower; }
        // return the value associated with the bin with the specified index
        float binAssignmentNumber(int index) const;
        bool equals(const BinMap&) const;
        bool operator<(const BinMap&) const;

        friend bool operator==(const BinMap&, const BinMap&);
    private:
        unsigned int _nBins;
        float _lower;
        float _width;
};

} // namespace lofar
} // namespace pelican
#endif // BINMAP_H 
