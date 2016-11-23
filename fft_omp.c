#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.1415926535
#endif

void fftOmp_initWtable(double complex *W, int W_size);
void fftUtils_reverseArray(double complex* array, int array_size);
double fftOmp_randomGen (double *seed);
int reverse(int N, int n);

void fft(double complex* array, unsigned long array_size)
{
  FILE *fp;
  double complex *W;
  unsigned long n = 1, i;
  unsigned long a = array_size / 2;

  // Create file
  fp = fopen("results.txt", "w");

  // Alocate and init W
  W = (double complex*) malloc( array_size/2 * sizeof(complex double));
  fftOmp_initWtable(W, array_size);

  // Bits Inversion
  fftUtils_reverseArray(array, 8);

  // Executing main FFT algorithm
  printf("  Executing FFT... \n");
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

  for (int i = 0; i < array_size; i++) {
    fprintf(fp, "Samples[%i] = %.2f + %.2fi\n", i, creal(array[i]), cimag(array[i]));
  }

  fclose(fp);
  free(W);
}

int main(int argc, char const *argv[]) {

  double complex samples[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  double complex *input;
  int exponent, i;
  int nThreads, tid, nUsedThreads;
  unsigned long n;
  static double seed;


  printf(" ********************************* \n");
  printf(" *** FFT - OMP implentation  ***** \n");
  printf(" ********************************* \n");
  printf(" ***  T.Cejrowski 21.11.2016   *** \n");
  printf(" ********************************* \n");
  printf ( "\n" );
  printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );
  printf ( "\n");
  printf ( "  Enter number of threads you want to use: \n" );
  scanf ( "\t %i", &nUsedThreads);

  omp_set_num_threads(nUsedThreads);
  #pragma omp parallel
  {
    #pragma omp master
    printf("  Using %i threads.\n", omp_get_num_threads());
  }

  printf ( "  Preparing for samples generation... Enter n (2^n) : \n" );
  scanf ( "\t %i", &exponent);
  n = pow(2, exponent);
  printf ( "  Generating %lu (2^%i) samples... \n\n",  n, exponent);

  seed  = 321.0;

  input = ( double complex * ) malloc ( n * sizeof ( double complex) );
  // Initialize data
#pragma omp parallel shared (input , seed) private (i)
#pragma omp for
  for (i = 0; i < n; i++ )
  {
    input[i] = fftOmp_randomGen(&seed);
    input[i] += fftOmp_randomGen(&seed)*I;
  }
  printf("  Done! \n");
  // Main FFT function
  fft(input, n);

  printf("  Success! Results are in results.txt file\n");

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
/******************************************************************************/

int reverse(int N, int n)
{
  int j, p = 0;
  for(j = 1; j <= log2(N); j++) {
    if(n & (1 << ((int) log2(N) - j)))
      p |= 1 << (j - 1);
  }
  return p;
}
/******************************************************************************/

void fftUtils_reverseArray(double complex* array, int array_size)
{
  double complex temp;
  int reversion_idx, i;

#pragma omp parallel shared ( array, array_size) private (i)
#pragma omp for
  for(i = 0; i < array_size/2; i++)
  {
    reversion_idx = reverse(array_size, i);

    temp = array[i];
    array[i] = array[reversion_idx];
    array[reversion_idx] = temp;
  }

}
