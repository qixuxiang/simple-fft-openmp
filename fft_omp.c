#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

#include "fft_utils.h"

#define DEBUG

void init_W_table(double complex *W, int W_size);


void fft(double complex* array, int array_size)
{
  double complex *W;
  int n = 1, i;
  int a = array_size / 2;

  // Alocate and init W
  W = (double complex*) malloc( array_size/2 * sizeof(complex double));
  init_W_table(W, array_size);

  // Bits Inversion
  reverse_array(array, 8);

  // For through stages
  for(int j = 0; j < log2(array_size); j++) {
  // Main loop paralelization
 #pragma omp parallel shared ( array, array_size, W, n, a ) private (i)
 #pragma omp for nowait
    for(i = 0; i < array_size; i++) {
      if(!(i & n)) {
         double complex temp_first_component = array[i];
         double complex temp_second_component = W[(i * a) % (n * a)]* array[i + n];

         array[i] = temp_first_component + temp_second_component;
         array[i + n] = temp_first_component - temp_second_component;
      }
    }
    n *= 2;
    a = a / 2;
  }

  #ifdef DEBUG
    for (int i = 0; i < array_size; i++) {
      printf("samples[%i] = %.2f + %.2fi\n", i, creal(array[i]), cimag(array[i]));
    }
  #endif

}

int main(int argc, char const *argv[]) {

  double complex samples[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int nThreads, tid;

  printf(" ********************************* \n");
  printf(" *** FFT - serial implentation *** \n");
  printf(" ********************************* \n");
  printf(" ***  T.Cejrowski 21.11.2016   *** \n");
  printf(" ********************************* \n");
  printf ( "\n" );
  printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
  printf ( "\n");

  fft(samples, 8);

  printf(" Success! \n");
  return 0;
}

void init_W_table(double complex *W, int number_of_samples)
{
  int i;

  W[0] = 1;
  W[1] = cexp(-2*M_PI*I/number_of_samples);

#pragma omp parallel shared ( W ) private ( i )
#pragma omp for nowait
  for(int i = 2; i < number_of_samples/2; i++) {
      W[i] = cpow(W[1], (double complex)i);
  }
}
