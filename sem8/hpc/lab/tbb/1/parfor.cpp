// Illustrates parallel for in TBB.
//
// HPC course, MIM UW
// Krzysztof Rzadca, LGPL

#include "tbb/tbb.h"
#include <iostream>
#include <math.h>


bool is_prime(long num) {
    long sqrt_num = (long) sqrt(num);
    for (long div = 2; div <= sqrt_num; ++div) {
        if ((num % div) == 0)
            return false;
    }
    return true;
}


int main() {
    const long limit = 5'000'000;
    tbb::tick_count seq_start_time = tbb::tick_count::now();
    for (long i = 1; i < limit; ++i) {
        is_prime(i);
    }
    tbb::tick_count seq_end_time = tbb::tick_count::now();
    double seq_time = (seq_end_time-seq_start_time).seconds();
    std::cout << "seq time for "<< limit <<" " << seq_time << "[s]" <<std::endl;

    tbb::tick_count par_start_time = tbb::tick_count::now();
    tbb::parallel_for(  // execute a parallel for
        tbb::blocked_range<long>(1, limit),  // pon a range from 1 to limit
        [](const tbb::blocked_range<long>& r) { // inside a loop, for a partial range r,
            // run this lambda
            for (long i = r.begin(); i != r.end(); ++i)
                is_prime(i);
        }
        );
    tbb::tick_count par_end_time = tbb::tick_count::now();
    double par_time = (par_end_time-par_start_time).seconds();
    std::cout << "par time for "<< limit <<" " << par_time << "[s]" <<std::endl;
}
