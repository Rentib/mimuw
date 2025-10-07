/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include <iostream>
#include <mpi.h>

static int const MPI_BACK_MESSAGE_TAG = 1;
static int const MPI_FRONT_MESSAGE_TAG = 2;

int main(int argc, char *argv[]) {
    int myProcessNo;
    int numProcesses;
    int prevProcessData;
    int nextProcessData;
    int prevProcessNo;
    int nextProcessNo;

    MPI_Request requests[4];
    MPI_Status statuses[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myProcessNo);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    prevProcessNo = myProcessNo > 0 ? myProcessNo - 1 : numProcesses - 1;
    nextProcessNo = myProcessNo < numProcesses - 1 ? myProcessNo + 1 : 0;

    MPI_Irecv(
            &prevProcessData,
            1,
            MPI_INT,
            prevProcessNo,
            MPI_BACK_MESSAGE_TAG,
            MPI_COMM_WORLD,
            &requests[0]
    );
    MPI_Irecv(
            &nextProcessData,
            1,
            MPI_INT,
            nextProcessNo,
            MPI_FRONT_MESSAGE_TAG,
            MPI_COMM_WORLD,
            &requests[1]
    );
    MPI_Isend(
            &myProcessNo,
            1,
            MPI_INT,
            prevProcessNo,
            MPI_FRONT_MESSAGE_TAG,
            MPI_COMM_WORLD,
            &requests[2]
    );
    MPI_Isend(
            &myProcessNo,
            1,
            MPI_INT,
            nextProcessNo,
            MPI_BACK_MESSAGE_TAG,
            MPI_COMM_WORLD,
            &requests[3]
    );
    MPI_Waitall(4, requests, statuses);

    std::cout << "Process "
        << myProcessNo
        << " received "
        << prevProcessData
        << " from the previous process and "
        << nextProcessData
        << " from the next process."
        << std::endl;

    MPI_Finalize();
    return 0;
}
