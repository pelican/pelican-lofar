#include "PolyphaseCoefficients.h"
#include <QtCore/QFile>
#include <QtCore/QTextStream>
#include <QtCore/QStringList>
#include <iostream>

namespace pelican {
namespace lofar {

/**
 * @details
 * Loads coefficients from matlab coefficient dump file written using
 * the matlab function dlmwrite().
 *
 * @param fileName
 * @param nFilterTaps
 * @param nChannels
 */
void PolyphaseCoefficients::load(const QString& fileName,
		const unsigned nFilterTaps, const unsigned nChannels)
{
	QFile file(fileName);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		throw QString("PolyphaseCoefficients::load(): "
				"Unable to open coefficients file %1.").arg(fileName);
	}

	resize(nFilterTaps, nChannels);

	QTextStream in(&file);

	for (unsigned c = 0; c < nChannels; ++c) {
		if (in.atEnd()) {
			throw QString("PolyphaseCoefficients::load(): "
					"Unexpectedly reached end of file.");
		}
		QString line = in.readLine();
//		std::cout << line.toStdString() << std::endl;
		QStringList chanCoeff = line.split(" ");
		if (chanCoeff.size() != nFilterTaps) {
			throw QString("PolyphaseCoefficients::load(): "
					"File format error. %1 %2").arg(nFilterTaps)
					.arg(chanCoeff.size());
		}
		for (unsigned t = 0; t < nFilterTaps; ++t) {
			_coeff[c * nFilterTaps + t] = chanCoeff.at(t).toDouble();
		}
	}
	file.close();
}

}// namespace lofar
}// namespace pelican

