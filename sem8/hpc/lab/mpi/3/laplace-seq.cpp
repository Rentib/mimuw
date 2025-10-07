/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <iomanip>
#include <vector>
#include <tuple>
#include "laplace-common.h"

#define OPTION_VERBOSE "--verbose"


static void printUsage(char const* progName) {
    std::cerr << "Usage:" << std::endl <<
                    "    " << progName << " [--verbose] <N>" << std::endl <<
                    "Where:" << std::endl <<
                    "   <N>         The number of points in each dimension (at least 2)." << std::endl <<
                    "   " << OPTION_VERBOSE << "   Prints the input and output systems." << std::endl;
}

static void freePointRow(const double* currentPointRow) {
    if (currentPointRow != nullptr) {
        delete(currentPointRow);
    }
}

static double* allocateCurrentPointRow(int numPointsPerDimension) {
    return new double[numPointsPerDimension];
}

static void freePoints(double** points, int numPointsPerDimension) {
    if (points != nullptr) {
        for (int i = 0; i < numPointsPerDimension; ++i) {
            freePointRow(points[i]);
        }

        delete(points);
    }
}

static double** allocatePoints(int numPointsPerDimension) {
    auto points = new double*[numPointsPerDimension];

    for (int i = 0; i < numPointsPerDimension; ++i) {
        points[i] = nullptr;
    }

    for (int i = 0; i < numPointsPerDimension; ++i) {
        points[i] = allocateCurrentPointRow(numPointsPerDimension);

        if (points[i] == nullptr) {
            freePoints(points, numPointsPerDimension);
            return nullptr;
        }
    }

    return points;
}

static void initializePoints(double **points, int numPointsPerDimension) {
    for (int i = 0; i < numPointsPerDimension; ++i) {
        for (int j = 0; j < numPointsPerDimension; ++j) {
            points[i][j] = Utils::getInitialValue(i, j, numPointsPerDimension);
        }
    }
}

static void printPoints(double **points, int numPointsPerDimension) {
    for (int i = 0; i < numPointsPerDimension; ++i) {
        std::cout << std::fixed << std::setprecision(10) << points[i][0];

        for (int j = 1; j < numPointsPerDimension; ++j) {
            std::cout << " " << std::fixed << std::setprecision(10) << points[i][j];
        }

        std::cout << std::endl;
    }
}

static InputOptions parseInput(int argc, char * argv[]) {
    int numPointsPerDimension = 0;
    bool verbose = false;
    int errorCode = 0;

    if (argc < 2) {
        std::cerr << "ERROR: Too few arguments!" << std::endl;
        printUsage(argv[0]);
        errorCode = 1;
    } else if (argc > 3) {
        std::cerr << "ERROR: Too many arguments!" << std::endl;
        printUsage(argv[0]);
        errorCode = 2;
    } else {
        int argIdx = 1;

        if (argc == 3) {
            if (strncmp(argv[argIdx], OPTION_VERBOSE, strlen(OPTION_VERBOSE)) != 0) {
                std::cerr << "ERROR: Unexpected option '" << argv[argIdx] << "'!" << std::endl;
                printUsage(argv[0]);
                errorCode = 3;
            }

            verbose = true;
            ++argIdx;
        }

        numPointsPerDimension = std::strtol(argv[argIdx], nullptr, 10);

        if (numPointsPerDimension < 2) {
            fprintf(stderr, "ERROR: The number of points, '%s', should be "
                            "a numeric value greater than or equal to 2!\n", argv[argIdx]);
            printUsage(argv[0]);
            errorCode = 4;
        }
    }

    return {numPointsPerDimension, verbose, errorCode};
}

std::tuple<int, double> performAlgorithm(double** points, double omega, double epsilon, int numPointsPerDimension) {
    double maxDiff;
    int numIterations = 0;

    do {
        maxDiff = 0.0;

        for (int color = 0; color < 2; ++color) {
            for (int i = 1; i < numPointsPerDimension - 1; ++i) {
                for (int j = 1 + (i % 2 == color ? 1 : 0); j < numPointsPerDimension - 1; j += 2) {
                    double tmp = (points[i - 1][j] + points[i + 1][j] + points[i][j - 1] + points[i][j + 1]) / 4.0;
                    double prev = points[i][j];

                    points[i][j] = (1.0 - omega) * points[i][j] + omega * tmp;
                    double diff = fabs(prev - points[i][j]);

                    if (diff > maxDiff) {
                        maxDiff = diff;
                    }
                }
            }
        }
        ++numIterations;
    }
    while (maxDiff > epsilon);

    return std::make_tuple(numIterations, maxDiff);
}


int main(int argc, char * argv[]) {
    struct timeval startTime {};
    struct timeval endTime {};

    auto inputOptions = parseInput(argc, argv);

    if (inputOptions.getErrorCode() != 0) {
        return inputOptions.getErrorCode();
    }

    auto numPointsPerDimension = inputOptions.getNumPointsPerDimension();
    auto isVerbose = inputOptions.isVerbose();

    double omega = Utils::getRelaxationFactor(numPointsPerDimension);
    double epsilon = Utils::getToleranceValue(numPointsPerDimension);
    auto pointsPointer = allocatePoints(numPointsPerDimension);

    if (pointsPointer == nullptr) {
        freePoints(pointsPointer, numPointsPerDimension);
        std::cerr << "ERROR: Memory allocation failed!" << std::endl;
        return 5;
    }

    initializePoints(pointsPointer, numPointsPerDimension);

    if (gettimeofday(&startTime, nullptr)) {
        freePoints(pointsPointer, numPointsPerDimension);
        std::cerr << "ERROR: Gettimeofday failed!" << std::endl;
        return 6;
    }

    /* Start of computations. */

    auto result = performAlgorithm(pointsPointer, omega, epsilon, numPointsPerDimension);

    /* End of computations. */

    if (gettimeofday(&endTime, nullptr)) {
        freePoints(pointsPointer, numPointsPerDimension);
        std::cerr << "ERROR: Gettimeofday failed!" << std::endl;
        return 7;
    }

    double duration =
            ((double)endTime.tv_sec + ((double)endTime.tv_usec / 1000000.0)) -
            ((double)startTime.tv_sec + ((double)startTime.tv_usec / 1000000.0));

    std::cerr << "Statistics: duration(s)="
            << std::fixed
            << std::setprecision(10)
            << duration << " #iters="
            << std::get<0>(result)
            << " diff="
            << std::get<1>(result)
            << " epsilon="
            << epsilon
            << std::endl;


    if (isVerbose) {
        printPoints(pointsPointer, numPointsPerDimension);
    }

    freePoints(pointsPointer, numPointsPerDimension);
    return 0;
}
