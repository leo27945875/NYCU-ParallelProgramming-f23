#include <stdio.h>
#include <time.h>
#include <thread>

#include "CycleTimer.h"

typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
} WorkerArgs;

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs *const args)
{

    // TODO FOR PP STUDENTS: Implement the body of the worker
    // thread here. Each thread could make a call to mandelbrotSerial()
    // to compute a part of the output image. For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.
    // Of course, you can copy mandelbrotSerial() to this file and 
    // modify it to pursue a better performance.

    int nowRow = args->threadId;
    while (nowRow < static_cast<int>(args->height)){
            mandelbrotSerial(
            args->x0, args->y0, args->x1, args->y1, 
            args->width, args->height, 
            nowRow, 1, 
            args->maxIterations, 
            args->output
        );
        nowRow += args->numThreads;
    }
    

    // int startRow, numRows;
    // numRows  = args->height / args->numThreads;
    // startRow = args->threadId * numRows;
    // if (args->threadId == args->numThreads - 1)
    //     numRows = args->height - startRow;
    
    // timespec t_start = {0, 0};
    // timespec t_end   = {0, 0};

    // clock_gettime(CLOCK_REALTIME, &t_start);
    // mandelbrotSerial(
    //     args->x0, args->y0, args->x1, args->y1, 
    //     args->width, args->height, 
    //     startRow, numRows, 
    //     args->maxIterations, 
    //     args->output
    // );
    // clock_gettime(CLOCK_REALTIME, &t_end);
    // printf("\nTime for thread %d is %ld (ms).\n", args->threadId, ((t_end.tv_sec - t_start.tv_sec) * 1000 + (t_end.tv_nsec - t_start.tv_nsec) / 1000000));
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i = 0; i < numThreads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < numThreads; i++)
    {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }

    workerThreadStart(&args[0]);

    // join worker threads
    for (int i = 1; i < numThreads; i++)
    {
        workers[i].join();
    }
}
