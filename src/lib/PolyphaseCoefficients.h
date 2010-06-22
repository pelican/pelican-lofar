#ifndef POLYPHASE_COEFFICIENTS_H_
#define POLYPHASE_COEFFICIENTS_H_


#include "pelican/data/DataBlob.h"
#include <vector>
#include <complex>
#include <QtCore/QIODevice>

namespace pelican {
namespace lofar {


/**
 * @class T_PolyphaseCoefficients
 *
 * @brief
 * Container class to hold polyphase filter coefficients.
 *
 * @details
 */

template <class T>
class T_PolyphaseCoefficients : public DataBlob
{
	public:
		/// Constructs a polyphase coefficients object.
		T_PolyphaseCoefficients(const QString& type) : DataBlob(type) {
			_nTaps = 0; _nChannels = 0;
		}

		/// Constructs a polyphase coefficients object of the specfied dimensions.
		T_PolyphaseCoefficients(const unsigned nTaps,
		        const unsigned nChannels, const QString& type)
		: DataBlob(type) {
			resize(nTaps, nChannels);
		}

		/// Destroys the object.
		virtual ~T_PolyphaseCoefficients() {}

	public:
		/// Clears the coefficients.
		void clear()
		{
			_coeff.clear();
			_nTaps = 0; _nChannels = 0;
		}

		/// Resizes the coefficient vector for nTaps and nChannels.
		void resize(const unsigned nTaps, const unsigned nChannels) {
			_nTaps = nTaps; _nChannels = nChannels;
			_coeff.resize(_nTaps * _nChannels);
		}

	public: // Accessor methods.
		/// Returns the number of coefficients.
		unsigned size() const { return _coeff.size(); }

		/// Returns the number of filter taps.
		unsigned nTaps() const { return _nTaps; }

		/// Returns the number of channels.
		unsigned nChannels() const { return _nChannels; }

		/// Returns a pointer to the vector of coefficients.
		T* coefficients() { return _coeff.size() > 0 ? &_coeff[0] : NULL; }

		/// Returns a pointer to the vector of coefficients (const overload).
		const T* coefficients() const {
			return _coeff.size() > 0 ? &_coeff[0] : NULL;
		}

	protected:
		std::vector<T> _coeff;
		unsigned _nTaps;
		unsigned _nChannels;
};


/**
 * @class PolyphaseCoefficients
 *
 * @brief
 * Container class to hold polyphase filter coefficients.
 *
 * @details
 * Template specialisation for complex double type.
 */

class PolyphaseCoefficients : public T_PolyphaseCoefficients<double>
{
	public:
		/// Constructs an empty polyphase filter coefficient data blob.
		PolyphaseCoefficients() : T_PolyphaseCoefficients<double>
		("PolyphaseCoefficients") {}

		/// Constructs a polyphase filter coefficient data blob.
		PolyphaseCoefficients(const unsigned nTaps, const unsigned nChannels) :
			T_PolyphaseCoefficients<double>(nTaps, nChannels,
			        "PolyphaseCoefficients") {}

		/// Constructs a polyphase filter coefficient data blob loading values
		/// the specified file.
		PolyphaseCoefficients(const QString& fileName, const unsigned nTaps,
				const unsigned nChannels) :
			T_PolyphaseCoefficients<double>("PolyphaseCoefficients")
		{
			load(fileName, nTaps, nChannels);
		}


	public:
		/// Load coefficients from matlab coefficient dump.
		void load(const QString& fileName, const unsigned nFilterTaps,
				const unsigned nChannels);
};

PELICAN_DECLARE_DATABLOB(PolyphaseCoefficients)

}// namespace lofar
}// namespace pelican

#endif // POLYPHASE_COEFFICIENTS_H_
