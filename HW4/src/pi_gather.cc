#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const double  reci = 1. / (double)RAND_MAX;
    unsigned int  seed = (unsigned int)(time(NULL) + world_rank);
    long long int iter = tosses / world_size;
    long long int hit  = 0;

    double x, y;
    for (long long int i = 0; i < iter; i++){
        x = ((double)rand_r(&seed) * reci) * 2. - 1.;
        y = ((double)rand_r(&seed) * reci) * 2. - 1.;
        if (x * x + y * y <= 1.)
            hit++;
    }

    // TODO: use MPI_Gather
    long long int* hits;
    if (world_rank == 0)
        hits = new long long int[world_size];
    
    MPI_Gather(&hit, 1, MPI_LONG_LONG, hits, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    hit = 0;
    for (int i = 0; i < world_size; i++)
        hit += hits[i];

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4. * hit / tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}
