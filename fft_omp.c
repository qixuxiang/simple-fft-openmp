#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

#include "fft_utils.h"

//#define DEBUG

void fftOmp_initWtable(double complex *W, int W_size);
double fftOmp_randomGen (double *seed);

void fft(double complex* array, unsigned long array_size)
{
  double complex *W;
  unsigned long n = 1, i;
  unsigned long a = array_size / 2;

  // Alocate and init W
  W = (double complex*) malloc( array_size/2 * sizeof(complex double));
  fftOmp_initWtable(W, array_size);

  // Bits Inversion
  fftUtils_reverseArray(array, 8);

  // For through stages
  for(unsigned long j = 0; j < log2(array_size); j++) {
  // Main loop paralelization
 #pragma omp parallel shared ( array, array_size, W, n, a ) private (i)
 #pragma omp for
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

  free(W);
}

int main(int argc, char const *argv[]) {

  double complex samples[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  double complex *input;

  int exponent;
  unsigned long n;

  static double seed;
  int nThreads, tid;

  printf(" ********************************* \n");
  printf(" *** FFT - OMP implentation  ***** \n");
  printf(" ********************************* \n");
  printf(" ***  T.Cejrowski 21.11.2016   *** \n");
  printf(" ********************************* \n");
  printf ( "\n" );
  printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
  printf ( "\n");

  printf ( "  Preparing for samples generation... Enter n (2^n) : \n" );
  scanf ( "\t %i", &exponent);
  n = pow(2, exponent);
  printf ( "  Generating %lu (2^%i) samples... \n\n",  n, exponent);

  seed  = 321.0;

  input = ( double complex * ) malloc ( n * sizeof ( double complex) );
  // Initialize data
  for (int i = 0; i < n; i++ )
  {
    input[i] = fftOmp_randomGen(&seed);
    input[i] += fftOmp_randomGen(&seed)*I;
  }
  printf(" Done! \n");
  // Main FFT function
  fft(input, n);

  printf(" Success! \n");
  free(input);

  return 0;
}
/******************************************************************************/

double fftOmp_randomGen (double *seed)
{
  double d2 = 0.2147483647e10;
  double t;
  double value;

  t = ( double ) *seed;
  t = fmod ( 16807.0 * t, d2 );
  *seed = ( double ) t;
  value = ( double ) ( ( t - 1.0 ) / ( d2 - 1.0 ) );

  return value;
}
/******************************************************************************/

void fftOmp_initWtable(double complex *W, int number_of_samples)
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
