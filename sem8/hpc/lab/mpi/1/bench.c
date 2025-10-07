#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int rank;
int size;

void bench(int bytes)
{
	char *buffer = (char *)malloc(bytes);
	if (buffer == NULL)
		abort();
	memset(buffer, 0, bytes);

	const int trials = 30;
	const int partner = (rank + 1) % 2;

	for (int i = 0; i < trials; i++) {
		MPI_Barrier(MPI_COMM_WORLD);
		double startTime, endTime;

		if (rank == 0) {
			startTime = MPI_Wtime();
			MPI_Send(buffer, bytes, MPI_BYTE, partner, 0,
				 MPI_COMM_WORLD);
			MPI_Recv(buffer, bytes, MPI_BYTE, partner, 0,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			endTime = MPI_Wtime();

			double rtt = (endTime - startTime);
			printf("%d, %d, %f\n", i, bytes, rtt);
		} else {
			MPI_Recv(buffer, bytes, MPI_BYTE, partner, 0,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(buffer, bytes, MPI_BYTE, partner, 0,
				 MPI_COMM_WORLD);
		}
	}

	free(buffer);
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv); /* initialize the library with parameters caught
				   by the runtime */

	int ns[] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000};
	int ns_count = sizeof(ns) / sizeof(ns[0]);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size != 2)
		abort();

	for (int i = 0; i < ns_count; i++) {
		bench(ns[i]);
	}

	MPI_Finalize(); /* mark that we've finished communicating */

	return 0;
}
