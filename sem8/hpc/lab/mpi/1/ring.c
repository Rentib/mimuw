#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size < 2)
		return 1;

	int next = (rank + 1 + size) % size;
	int prev = (rank - 1 + size) % size;

	MPI_Barrier(MPI_COMM_WORLD);

	uint64_t value = 1;
	if (rank == 0) {
		MPI_Send(&value, 1, MPI_UINT64_T, next, 0, MPI_COMM_WORLD);
		MPI_Recv(&value, 1, MPI_UINT64_T, prev, 0, MPI_COMM_WORLD,
			 MPI_STATUS_IGNORE);

		printf("%zu\n", value);
	} else {
		MPI_Recv(&value, 1, MPI_UINT64_T, prev, 0, MPI_COMM_WORLD,
			 MPI_STATUS_IGNORE);
		value *= rank + 1;
		MPI_Send(&value, 1, MPI_UINT64_T, next, 0, MPI_COMM_WORLD);
	}
	MPI_Finalize();

	return 0;
}
