#include <iostream>

int main(){

    long long nToss = 100000000;
    long long nCircle = 0;

    const double max = static_cast<double>(RAND_MAX);

    double x, y;

    for (long long i = 0; i <= nToss; i++){
        x = (rand() / max - 0.5) * 2.;
        y = (rand() / max - 0.5) * 2.;
        if (x * x + y * y <= 1.)
            nCircle++;
    }

    std::cout << "Pi = " << 4. * nCircle / static_cast<double>(nToss) << std::endl;

    return 0;
}