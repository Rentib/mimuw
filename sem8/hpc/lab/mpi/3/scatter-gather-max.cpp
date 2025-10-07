/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include <iostream>
#include <mpi.h>

static int const NUM_NUMBERS_PER_PROCESS = 10;
static int const ROOT_PROCESS = 0;

static int computeMax(int const arr[], int num) {
    int max = arr[0];
    for (int i = 1; i < num; ++i) {
        if (max < arr[i]) {
            max = arr[i];
        }
    }
    return max;
}


int main(int argc, char *argv[]) {
    int myProcessNo;
    int numProcesses;
    int *allNumbers;
    int *partialResults;
    int maxNumber;
    int printNumbers = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myProcessNo);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    if (argc > 1) {
        printNumbers = 1;
    }
    auto myNumbers = new int[NUM_NUMBERS_PER_PROCESS];
    
    if (myProcessNo == ROOT_PROCESS) {
        allNumbers = new int[NUM_NUMBERS_PER_PROCESS * numProcesses];
        partialResults = new int[numProcesses];

        for (int i = 0; i < NUM_NUMBERS_PER_PROCESS * numProcesses; ++i) {
            allNumbers[i] = rand() % (NUM_NUMBERS_PER_PROCESS * numProcesses);
        }
        if (printNumbers) {
            std::cout << "The numbers:";
            for (int i = 0; i < NUM_NUMBERS_PER_PROCESS * numProcesses; ++i) {
                std::cout << " " << allNumbers[i];
            }
            std::cout << "." << std::endl;
        }
    } else {
        allNumbers = nullptr;
        partialResults = nullptr;
    }
    MPI_Scatter(
            allNumbers,
            NUM_NUMBERS_PER_PROCESS,
            MPI_INT,
            myNumbers,
            NUM_NUMBERS_PER_PROCESS,
            MPI_INT,
            ROOT_PROCESS,
            MPI_COMM_WORLD
    );
    maxNumber = computeMax(myNumbers, NUM_NUMBERS_PER_PROCESS);
    if (printNumbers) {
        std::cout << "The result of process " << myProcessNo << " is " << maxNumber << "." << std::endl;
    }
    MPI_Gather(
            &maxNumber,
            1 /* just one number */,
            MPI_INT,
            partialResults,
            1 /* one number per process */,
            MPI_INT,
            ROOT_PROCESS,
            MPI_COMM_WORLD
    );
    if (printNumbers) {
        /* The barrier is here only to ensure that */
        /* the final maximum is printed AFTER      */
        /* the partial result of each process      */
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (myProcessNo == ROOT_PROCESS) {
        maxNumber = computeMax(partialResults, numProcesses);

        if (printNumbers) {
            std::cout << "The final result: " << maxNumber << "." << std::endl;
        }

        delete[](partialResults);
        delete[](allNumbers);
    }
    delete[](myNumbers);
    MPI_Finalize();
    return 0;
}
