#include <complex>
#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string>
#include <vector>

#include "utils/bmp.cpp"

void compress(const uint32_t valuesCount, const int accuracy,
	      const uint8_t *values, float *Xreal, float *Ximag)
{
	// values, Xreal and Ximag are values describing single color of single
	// row of bitmap. This function will be called once per each (color,
	// row) combination.
	for (int k = 0; k < accuracy; k++) {
		for (uint i = 0; i < valuesCount; i++) {
			float theta = (2 * M_PI * k * i) / valuesCount;
			Xreal[k] += values[i] * cos(theta);
			Ximag[k] -= values[i] * sin(theta);
		}
	}
}

void decompress(const uint32_t valuesCount, const int accuracy, uint8_t *values,
		const float *Xreal, const float *Ximag)
{
	// values, Xreal and Ximag are values describing single color of single
	// row of bitmap. This function will be called once per each (color,
	// row) combination.
	std::vector<float> rawValues(valuesCount, 0);

	for (uint i = 0; i < valuesCount; i++) {
		for (int k = 0; k < accuracy; k++) {
			float theta = (2 * M_PI * k * i) / valuesCount;
			rawValues[i] +=
			    Xreal[k] * cos(theta) + Ximag[k] * sin(theta);
		}
		values[i] = rawValues[i] / valuesCount;
	}
}

void compressParNaive(const uint32_t valuesCount, const int accuracy,
		      const uint8_t *values, float *Xreal, float *Ximag)
{
#pragma omp parallel for shared(Xreal, Ximag)
	for (int k = 0; k < accuracy; k++) {
		for (uint32_t i = 0; i < valuesCount; i++) {
			float theta = (2 * M_PI * k * i) / valuesCount;
			float cos_theta = cos(theta);
			float sin_theta = sin(theta);
#pragma omp atomic
			Xreal[k] += values[i] * cos_theta;
#pragma omp atomic
			Ximag[k] -= values[i] * sin_theta;
		}
	}
}

void compressPar(const uint32_t valuesCount, const int accuracy,
		 const uint8_t *values, float *Xreal, float *Ximag)
{
#pragma omp parallel for schedule(static) shared(Xreal, Ximag)
	for (int k = 0; k < accuracy; k++) {
		float real = 0;
		float imag = 0;

#pragma omp simd reduction(+ : real, imag)
		for (uint i = 0; i < valuesCount; i++) {
			float theta = (2 * M_PI * k * i) / valuesCount;
			real += values[i] * cos(theta);
			imag += values[i] * sin(theta);
		}

		Xreal[k] += real;
		Ximag[k] -= imag;
	}
}

void decompressPar(const uint32_t valuesCount, const int accuracy,
		   uint8_t *values, const float *Xreal, const float *Ximag)
{
	std::vector<float> rawValues(valuesCount, 0);

#pragma omp parallel shared(rawValues, values)
	{
		int threadId = omp_get_thread_num();
		int numThreads = omp_get_num_threads();

		for (uint i = threadId; i < valuesCount; i += numThreads) {
			for (int k = 0; k < accuracy; k++) {
				float theta = (2 * M_PI * k * i) / valuesCount;
				rawValues[i] += Xreal[k] * cos(theta) +
						Ximag[k] * sin(theta);
			}
		}

#pragma omp barrier
		if (threadId == 0) {
			for (uint i = 0; i < valuesCount; i++)
				values[i] = rawValues[i] / valuesCount;
		}
	}
}

void runExperiments()
{
	float compressTime, decompressTime;
	float compressTimeParNaive, compressTimePar, decompressTimePar;
	double compressNaiveSpeedup, compressSpeedup, decompressSpeedup;
	std::vector<size_t> accuracyValues = {8, 16, 32};
	std::vector<size_t> numThreads = {2, 4, 8, 16, 32, 64};

	printf("Accuracy,Threads,CompressNaiveSpeedup,CompressSpeedup,"
	       "DecompressSpeedup\n");
	for (size_t accuracy : accuracyValues) {
		BMP bmp;
		bmp.read("example.bmp");
		compressTime = bmp.compress(compress, accuracy);
		decompressTime = bmp.decompress(decompress);

		for (size_t threads : numThreads) {
			BMP bmpPar;
			bmpPar.read("example.bmp");

			omp_set_num_threads(threads);

			compressTimeParNaive =
			    bmpPar.compress(compressParNaive, accuracy);
			decompressTimePar = bmpPar.decompress(decompressPar);
			compressTimePar =
			    bmpPar.compress(compressPar, accuracy);

			compressNaiveSpeedup =
			    compressTime / compressTimeParNaive;
			compressSpeedup = compressTime / compressTimePar;
			decompressSpeedup = decompressTime / decompressTimePar;

			printf("%8lu,%7lu,%20.4f,%15.4f,%17.4f\n", accuracy,
			       threads, compressNaiveSpeedup, compressSpeedup,
			       decompressSpeedup);
		}
	}
}

int main()
{
	BMP bmp, bmpPar;

	size_t accuracy = 16; // We are interested in values from range [8; 64]

	// bmp.{compress,decompress} will run provided function on every bitmap
	// row and color.
	bmp.read("example.bmp");
	bmp.compress(compress, accuracy);
	bmp.decompress(decompress);
	bmp.write("example_result.bmp");

	// parallel
    omp_set_num_threads(4);
	bmpPar.read("example.bmp");
	bmpPar.compress(compressParNaive, accuracy);
	bmpPar.decompress(decompressPar);
	bmpPar.write("example_result_par.bmp");

	// runExperiments();

	return 0;
}
