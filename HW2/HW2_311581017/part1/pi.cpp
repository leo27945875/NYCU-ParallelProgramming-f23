#include <iostream>
#include <string>
#include <pthread.h>


typedef long long int toss_t ;

struct ThreadInfo{
    toss_t n_toss;
};


pthread_mutex_t global_mutex;
toss_t          total_in_circle_toss = 0;


void* random_shoot(void* input_info){

    ThreadInfo*  info       = (ThreadInfo*)input_info;
    const double max        = (double)RAND_MAX;
    unsigned int local_seed = (unsigned int)time(NULL) + (unsigned int)pthread_self();

    toss_t  n_in_circle = 0;
    double  x, y;
    for (toss_t i = 0; i < info->n_toss; i++){
        x = (((double)rand_r(&local_seed)) / max - 0.5) * 2.;
        y = (((double)rand_r(&local_seed)) / max - 0.5) * 2.;
        if (x * x + y * y <= 1.)
            n_in_circle++;
    }

    pthread_mutex_lock(&global_mutex);
    total_in_circle_toss += n_in_circle;
    pthread_mutex_unlock(&global_mutex);

    pthread_exit(0);
}

int main(int argc, char** argv){
    int    thread_count          = std::stoi(argv[1]);
    toss_t toss_count            = std::stoi(argv[2]);
    toss_t toss_count_per_thread = toss_count / thread_count;

    pthread_t*      thread_ids = new pthread_t[thread_count];
    pthread_attr_t  attr;

    ThreadInfo* infos = new ThreadInfo[thread_count];
    for (int i = 0; i < thread_count; i++){
        if (i == thread_count - 1)
            infos[i].n_toss = toss_count_per_thread + (toss_count % thread_count);
        else
            infos[i].n_toss = toss_count_per_thread;

        pthread_create(&thread_ids[i], nullptr, &random_shoot, reinterpret_cast<void*>(infos + i));
    }
    for (int i = 0; i < thread_count; i++)
        pthread_join(thread_ids[i], nullptr);

    double PI = 4. * total_in_circle_toss / toss_count;
    printf("%f\n", PI);

    return 0;
}
