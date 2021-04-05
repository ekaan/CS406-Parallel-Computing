#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

using namespace std;


void usage()
{
  cout << "USAGE: ./exec <filename>" << endl;
  exit(0);
}

int main(int argc, const char** argv)
{

  if(argc != 2)
    usage();

  string line;

  const char* filename = argv[1];
  ifstream input (filename);
  if(input.fail())
    return 0;


  int N;
  int **M;
  getline(input,line);
  N = atoi(line.c_str());

  M = new int*[N];
  for(int i = 0; i < N; i ++)
    M[i] = new int[N];


  int linectr = 0;
  while(getline(input,line)){
    stringstream ss(line);
    int temp;
    int ctr = 0;
    while(ss >> temp)
      M[linectr][ctr++] = temp;

    linectr++;
  }

  double start, end;
  for(int t = 1; t <=16; t*=2) //16 OLACAK
  { //t is the number of threads                                                                                                 
    start = omp_get_wtime();
  ////YOUR CODE GOES HERE/////----------------------------------------------

  long long int tn11 = (long long int)((1 << N-1)); // ((1 << N-1) - 1);

  //TAKE THE TRANSPOSE------------------------------------------------------

  int ** M_T = new int* [N];

  for(int j = 0 ; j < N ; j++)
  {
    M_T[j] = new int [N];
    for(int k = 0 ; k < N ; k++)
    {
      M_T[j][k] = M[k][j];
    }
  }
//--------------------------------------------------------------------------
  long double p = 1;
  double lastColumn, t_sum;

  double* x = new double [N];

  for(long long int i = 0 ; i < N ; i++)
  {
    lastColumn = M[i][N-1]; //--without transpose
    //lastColumn = M_T[N-1][i];
    t_sum = 0;

    for(long long int j = 0 ; j < N ; j++)
      t_sum += M[i][j];

    x[i] = lastColumn - t_sum/2.0;
    p *= x[i];
  }
//--------------------------------------------------------------------------
#pragma omp parallel num_threads(t) proc_bind(spread)
{
//--------------------------------------------------------------------------
  double* x_spec = new double [N];

  for(int i = 0 ; i < N ; i++)
    {
      x_spec[i] = x[i];
    }
//--------------------------------------------------------------------------
  int th_id = omp_get_thread_num();
  long long int start_loc =  ((th_id * tn11) / t);
  long long int gray = (start_loc) ^ (start_loc >> 1);

  int ct = 0;
  while(gray)
  {
    if(gray & 1 == 1)
    {
      for(int k = 0 ; k < N ; k++)
        x_spec[k] += M[k][ct]; //without transpose
        //x_spec[k] += M_T[ct][k];
    }
    gray = gray >> 1;
    ct += 1;
  }
//--------------------------------------------------------------------------
#pragma omp for schedule(static) reduction(+:p)
  for(long long int i = 1 ; i < tn11 ; i++)                                                                                               
  {
    long long int y,z,j,c,s;
    long double pr_Sign;
    
    y = ((i) >> 1)^(i);
    j = ((i-1) >> 1)^(i-1);
    //(i == 0) ? z = __builtin_ctz((long double)(0 ^ y))+1 : z = __builtin_ctz((long double)(y ^ j))+1;
    z = __builtin_ctz(y ^ j);
    
    s = (((y & ( 1 << z )) >> z) << 1)-1;
    ((i) % 2 == 1) ? pr_Sign = -1 : pr_Sign = 1;

    long double prod_x = 1;
   // #pragma omp simd
    for(long long int j = 0 ; j < N ; j++)
    {
      x_spec[j] = x_spec[j] + M[j][z] * s; //for transpose
      //x_spec[j] = x_spec[j] + M_T[z][j] * s;
      prod_x *= x_spec[j];
    }
    p = p + pr_Sign * prod_x;
  }
  delete [] x_spec;
 }
 long long int result = (4 * (N & 1) - 2) * p;

  //// YOUR CODE ENDS HERE
 end = omp_get_wtime();
 cout << "Threads: " << t << "\tResult:" << result << "\tTime:" << end - start << " s" << endl;
  }
  return 0;

}
//PRAGMA OMP SIMD

/*
    long long int g_code = y;
    int ct = 0;
    while(g_code != 0)
    {
      if(g_code % 2 == 1)
      {
        for(int k = 0 ; k < N ; k++)
          x[k] += M[k][ct];
      }

      g_code = g_code >> 1;
      ct += 1;
    }
*/
/*
    int ct = 0;
    long long int y2 = 0;
    while(y != 0 && y2 != 0)
    {

      if ((y % 2 == 1) && (y2 % 2 == 0))
      {
        for(int k = 0 ; k < N ; k++)
          x[k] += M[k][ct];
      }
      else if ((y % 2 == 0) && (y2 % 2 == 1))
      {
        for(int k = 0 ; k < N ; k++)
          x[k] -= M[k][ct];
      }
      ct += 1;
      y = y >> 1;
    }
*/
/*
#pragma omp parallel for num_threads(t) schedule(static) reduction(+:p) private(z,s,pr_Sign,y,j) firstprivate(x)
  for(long long int i = 0 ; i < tn11 ; i++)                                                                                               
  {
    y = ((i+1) >> 1)^(i+1);
    j = ((i >> 1)^i);
    (i == 0) ? z = log2((long double)(0 ^ y))+1 : z = log2((long double)(y ^ j))+1;
    long long int c = ((y >> z-1) % 2) + 1;

    (c % 2 == 1) ? s = -1 : s = 1;
    ((i+1) % 2 == 1) ? pr_Sign = -1 : pr_Sign = 1;

    for(long long int j = 0 ; j < N ; j++)
    {
      x[j] = x[j] + M[j][z-1] * s;
    }

    long double  prod_x = 1;

    for(long long int j = 0 ; j < N ; j++)
    {
      prod_x *= x[j];
    }
    p += (pr_Sign * prod_x);
  }
  long long int result = (4 * (N % 2) - 2) * p;
  delete []  x;
  */
