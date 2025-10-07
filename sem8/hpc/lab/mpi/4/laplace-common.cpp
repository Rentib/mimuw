/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#import "laplace-common.h"
#import <cassert>
#import <mpi.h>
#import <iostream>
#include <iomanip>

#define LAPLACE_I0 1.56
#define LAPLACE_IN 10.85
#define LAPLACE_J0 5.32
#define LAPLACE_JN 3.88
#define LAPLACE_XX 0.0

double Utils::getInitialValue(int i, int j, int numPointsPerDimension) {
    if (i == 0) {
        return LAPLACE_I0;
    } else if (i == numPointsPerDimension - 1) {
        return LAPLACE_IN;
    } else if (j == 0) {
        return LAPLACE_J0;
    } else if (j == numPointsPerDimension - 1) {
        return LAPLACE_JN;
    } else {
        return LAPLACE_XX;
    }
}

double Utils::getRelaxationFactor(int numPointsPerDimension) {
    double p = cos(M_PI / (double)numPointsPerDimension);
    return 1.6 / (1 + sqrt(1 - p * p));
}

double Utils::getToleranceValue(int numPointsPerDimension) {
    return 0.000002 / (2.0 - 1.25 * getRelaxationFactor(numPointsPerDimension));
}

int InputOptions::getNumPointsPerDimension() {
    return this->numPointsPerDimension;
}

bool InputOptions::isVerbose() {
    return this->verbose;
}

int InputOptions::getErrorCode() {
    return this->errorCode;
}

int GridFragment::getFirstRowIdxOwnedByProcess(
        int numPointsPerDimension,
        int numProcesses,
        int rank
) {
    int numBase = numPointsPerDimension / numProcesses;
    int numInc = numPointsPerDimension % numProcesses;
    int idx = rank * numBase + (rank > numInc ? numInc : rank);
    return idx;
}

GridFragment::GridFragment(int numPointsPerDimension, int numProcesses, int myRank, bool debug) {
    int color, i;
    int maxPointsPerColorPerRow;
    int numTotalRows;

    assert(numPointsPerDimension >= numProcesses);
    for (color = 0; color < 2; ++color) {
        this->data[color] = nullptr;
    }

    this->extraRowForPrinting = nullptr;
    this->gridDimension = numPointsPerDimension;
    this->firstRowIdxIncl = this->getFirstRowIdxOwnedByProcess(
            numPointsPerDimension,
            numProcesses,
            myRank);
    this->lastRowIdxExcl = this->getFirstRowIdxOwnedByProcess(
            numPointsPerDimension,
            numProcesses,
            myRank + 1);
    numTotalRows = this->lastRowIdxExcl - this->firstRowIdxIncl + 2;
    maxPointsPerColorPerRow = (numPointsPerDimension + 1) / 2;

    if (debug) {
        std::cerr <<
                  "DBG PROCESS " << myRank
                  << ": GD="
                  << this->gridDimension
                  << " FR="
                  << this->firstRowIdxIncl
                  << " LR="
                  << this->lastRowIdxExcl
                  << " TR="
                  << numTotalRows
                  << " MPPCPR="
                  << maxPointsPerColorPerRow
                  << std::endl;
    }

    for (color = 0; color < 2; ++color) {
        this->data[color] = new double*[numTotalRows];
        if (this->data[color] == nullptr) {
            this->free();
        }
        for (i = 0; i < numTotalRows; ++i) {
            this->data[color][i] = nullptr;
        }
        for (i = 0; i < numTotalRows; ++i) {
            this->data[color][i] = new double[maxPointsPerColorPerRow]; 
            if (this->data[color][i] == nullptr) {
                this->free();
            }
        }
    }

    this->extraRowForPrinting = new double[numPointsPerDimension];
    if (this->extraRowForPrinting == nullptr) {
        this->free();
    }
}

void GridFragment::free() {
    int numOwnedRows = this->lastRowIdxExcl - this->firstRowIdxIncl + 2;
    for (int color = 0; color < 2; ++color) {
        if (this->data[color] != nullptr) {
            for (int i = 0; i < numOwnedRows; ++i) {
                if (this->data[color][i] != nullptr) {
                    delete(this->data[color][i]);
                }
            }
            delete(this->data[color]);
            this->data[color] = nullptr;
        }
    }

    if (this->extraRowForPrinting != nullptr) {
        delete(this->extraRowForPrinting);
        this->extraRowForPrinting = nullptr;
    }
    delete(this);
}

void GridFragment::printEntireGrid(
        int myRank,
        int numProcesses
) {
    if (myRank == 0) {
        MPI_Status status;
        int rowIdx = 0;
        for (rowIdx = this->firstRowIdxIncl; rowIdx < this->lastRowIdxExcl; ++rowIdx) {
            std::cout << std::fixed << std::setprecision(5) << GP(this, rowIdx, 0);
            for (int colIdx = 1; colIdx < this->gridDimension; ++colIdx) {
                std::cout << " " << std::fixed << std::setprecision(5) << GP(this, rowIdx, colIdx);
            }
            std::cout << std::endl;
        }
        for (int procIdx = 1; procIdx < numProcesses; ++procIdx) {
            int lastIdx = this->getFirstRowIdxOwnedByProcess(
                    this->gridDimension,
                    numProcesses,
                    procIdx + 1);
            for (; rowIdx < lastIdx; ++rowIdx) {
                MPI_Recv(
                        this->extraRowForPrinting,
                        this->gridDimension,
                        MPI_DOUBLE,
                        procIdx,
                        PRINT_MSG_TAG,
                        MPI_COMM_WORLD,
                        &status
                );
                std::cout << std::fixed << std::setprecision(5) << this->extraRowForPrinting[0];
                for (int colIdx = 1; colIdx < this->gridDimension; ++colIdx) {
                    std::cout << " " << std::fixed << std::setprecision(5) << this->extraRowForPrinting[colIdx];
                }
                std::cout << std::endl;
            }
        }
    } else {
        for (int rowIdx = this->firstRowIdxIncl; rowIdx < this->lastRowIdxExcl; ++rowIdx) {
            for (int colIdx = 0; colIdx < this->gridDimension; ++colIdx) {
                this->extraRowForPrinting[colIdx] = GP(this, rowIdx, colIdx);
            }
            MPI_Send(
                    this->extraRowForPrinting,
                    this->gridDimension,
                    MPI_DOUBLE,
                    0,
                    PRINT_MSG_TAG,
                    MPI_COMM_WORLD
            );
        }
    }
}

void GridFragment::initialize() {
    for (int rowIdx = this->firstRowIdxIncl; rowIdx < this->lastRowIdxExcl; ++rowIdx) {
        for (int colIdx = 0; colIdx < this->gridDimension; ++colIdx) {
            GP(this, rowIdx, colIdx) = Utils::getInitialValue(rowIdx, colIdx, this->gridDimension);
        }
    }
}

int GridFragment::getNumColorPointsInRow(
        int rowIdx,
        int color) {
    int base = this->gridDimension / 2;
    int remainder = this->gridDimension % 2;
    return base + remainder * (rowIdx % 2 == color ? 1 : 0);
}
