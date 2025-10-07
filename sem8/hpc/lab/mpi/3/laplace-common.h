/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#ifndef __LAPLACE_COMMON_H__
#define __LAPLACE_COMMON_H__

#include <cmath>

/**
 * A reference to the grid point with
 * coordinates (i, j). It is assumed that
 * row i belongs to fragment fp.
 */
#define GP(fp, i, j) ((fp)->data[((i) + (j)) % 2][(i) - (fp)->firstRowIdxIncl + 1][(j) / 2])

#define PRINT_MSG_TAG 543

class InputOptions {
private:
    int numPointsPerDimension;
    bool verbose;
    int errorCode;

public:
    InputOptions(int numPointsPerDimension, bool verbose, int errorCode) :
            numPointsPerDimension(numPointsPerDimension),
            verbose(verbose),
            errorCode(errorCode) {}

    int getNumPointsPerDimension();
    bool isVerbose();
    int getErrorCode();
};

class Utils {
public:
    static double getInitialValue(int i, int j, int numPointsPerDimension);
    static double getRelaxationFactor(int numPointsPerDimension);
    static double getToleranceValue(int numPointsPerDimension);
};

/**
 * We explicitly split data for the two colors.
 * Each buffer has enough space to hold all
 * points from a row with the same color.
 */
class GridFragment {
public:
    double **data[2];
    double *extraRowForPrinting;
    int gridDimension;
    int firstRowIdxIncl;
    int lastRowIdxExcl;

    GridFragment(int numPointsPerDimension,
                 int numProcesses,
                 int myRank,
                 bool debug = false);

    /**
     * Prints the entire grid.
     * For illustrative purposes, you may
     * view how this function operates.
     */
    void printEntireGrid(int myRank,
                         int numProcesses);

    /**
     * Returns the number of points of a given
     * color in a row with a given index.
     */
    int getNumColorPointsInRow(int rowIdx, int color);

    void initialize();
    void free();

private:
    static int getFirstRowIdxOwnedByProcess(int numPointsPerDimension,
                                            int numProcesses,
                                            int rank);
};

#endif /* __LAPLACE_COMMON_H__ */
