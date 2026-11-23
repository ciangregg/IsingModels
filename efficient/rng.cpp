#include <gsl/gsl_rng.h>
#include <cstdio>

int main() {
    gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);  // the MT algorithm
    gsl_rng_set(r, 12345);                         // seed

    for (int i = 0; i < 5; ++i)
        std::printf("%f\n", gsl_rng_uniform(r)*20);   // uniform in [0,1)

    gsl_rng_free(r);
    return 0;
}