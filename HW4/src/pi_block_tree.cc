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

    // TODO: binary tree redunction
    int           now_size   = world_size;
    int           rank_inter = 2;
    int           half_inter = 1;
    long long int recv_hit;
    MPI_Status    status;
    while (now_size > 1){
        if (world_rank % rank_inter != 0){
            MPI_Send(&hit, 1, MPI_LONG_LONG, world_rank - half_inter, 0, MPI_COMM_WORLD);
            break;
        }
        else if (world_rank + 1 < world_size){
            MPI_Recv(&recv_hit, 1, MPI_LONG_LONG, world_rank + half_inter, 0, MPI_COMM_WORLD, &status);
            hit += recv_hit;
        }

        if (now_size % 2 == 1){
            now_size >>= 1;
            now_size  += 1;
        } 
        else 
            now_size >>= 1;

        rank_inter <<= 1;
        half_inter <<= 1;
    }

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
