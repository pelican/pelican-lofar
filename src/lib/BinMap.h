#ifndef BINMAP_H
#define BINMAP_H

#include <QtCore/QHash>
#include <QtCore/QMap>

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

        /// reset the map with all default values
        void reset(unsigned int numberOfBins);

        /// set the starting value for the first bin
        void setStart(double);

        /// set the starting value for the last bin
        void setEnd(double);

        /// set the width of each bin
        void setBinWidth(double);

        int binIndex(double) const;
        double width() const { return _width; };
        unsigned int numberBins() const { return _nBins; };
        double startValue() const { return _lower; }
        double endValue() const { return _lower + _width*_nBins; }
        // the value represented by the left hand edge of the bin
        double binStart(unsigned int index) const;
        // the value represented by the right hand edge of the bin
        double binEnd(unsigned int index) const;
        // return the value associated with the bin with the specified index
        inline double binAssignmentNumber(int index) const {
            return _lower + ( _width * index );
        }
        bool operator<(const BinMap&) const;

        // a representation for hashing
        unsigned int hash() const;

        friend bool operator==(const BinMap&, const BinMap&);
    private:
        static QMap< unsigned int, QMap< double, QMap<double, unsigned int> > > _unique; // assigns unique id to a BinMap type
        static unsigned int _uniqueCount; // guarantees a unique id
        unsigned int _nBins;
        double _lower;
        double _width;
        double _halfwidth;
        mutable unsigned int _hash;
};

unsigned int qHash(const BinMap& key);
} // namespace lofar
} // namespace pelican
#endif // BINMAP_H
