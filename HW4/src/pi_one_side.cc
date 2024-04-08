#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>


bool isAnyZero(long long int* hits, int size){
    for (int i = 0; i < size; i++)
        if (hits[i] == 0)
            return true;
    return false;
}


int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

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

    if (world_rank == 0)
    {
        // Master
        long long int* hits = new long long int[world_size];
        memset(hits, 0, world_size * sizeof(long long int));
        hits[0] = hit;

        MPI_Win_create(hits, world_size * sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        bool is_not_finish = true;
        while (is_not_finish){
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            is_not_finish = isAnyZero(hits, world_size);
            MPI_Win_unlock(0, win);
        }

        hit = 0;
        for (int i = 0; i < world_size; i++)
            hit += hits[i];
    }
    else
    {
        // Workers
        MPI_Win_create(nullptr, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&hit, 1, MPI_LONG_LONG, 0, world_rank, 1, MPI_LONG_LONG, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
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