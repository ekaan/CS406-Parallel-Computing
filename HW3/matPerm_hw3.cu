#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

using namespace std;



#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
  if(e!=cudaSuccess) {                                              \
     printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
        exit(0); \
	 }                                                                 \
	 }
  
void usage()
{
  cout << "USAGE: ./exec <filename>" << endl;
  exit(0);
}

__global__
void perMat (double* x, double* x_pre_d, int rowSize, long long* myMat, double* p, long long tNum) {
  unsigned long long tn11 = 1LL << (rowSize-1);
  unsigned long long chunkSize = tn11 / tNum;
  
  unsigned long long t_Id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long start_loc = t_Id * chunkSize + 1;
  unsigned long long end_loc = min(start_loc + chunkSize, tn11);

  double *my_x =  x + t_Id*rowSize;
  for (int i = 0 ; i < rowSize ; i++) {
    my_x[i] = x_pre_d[i];
  }

  long long gray = (start_loc - 1) ^ ((start_loc-1) >> 1);
  int ct = 0;
  while(gray) {
    if(gray & 1LL) {
      for(int k = 0 ; k < rowSize ; k++) {
        my_x[k] += myMat[ct*rowSize + k];
      }
    }
    gray = gray >> 1;
    ct += 1;
  }
  
  
  double local_p = 0;

  if(t_Id < 16) {
    printf("%d, %d, %d, %ld, %ld\n",  blockIdx.x, blockDim.x, threadIdx.x, start_loc, end_loc);
  }
  
  int pr_Sign = 1; if(start_loc % 2 == 1) pr_Sign = -1;  
  for (long long i = start_loc; i < end_loc; i++) {
    unsigned long long prev_gray, bit_id;
    
    gray = (i >> 1) ^ i;
    prev_gray = ((i-1) >> 1) ^ (i-1);
    bit_id = __ffs(gray ^ prev_gray) - 1;// -1

    int s = -1;
    if(gray & (1LL << bit_id)) {
      s = 1;
    }
        
    double prod_x = 1;
    for(int j = 0 ; j < rowSize ; j++) {
      my_x[j] += myMat[bit_id * rowSize + j] * s;
      prod_x *= my_x[j];
    }
    local_p += pr_Sign * prod_x;
    pr_Sign *= -1;
  }
  atomicAdd(p, local_p);
}

int main(int argc, const char** argv)
{

  if(argc != 3)
    usage();

  string line;

  const char* filename = argv[1];
  int devId = atoi(argv[2]);

  ifstream input (filename);
  if(input.fail())
    return 0;


  long long N;
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
  ////YOUR CODE GOES HERE/////----------------------------------------------

  cudaSetDevice(devId); // SET DEVICE

  long long tn11 = (long long)(( (long long)1 << (N-1) ));

  cout << "tn11: " << tn11 << endl;

  int ** M_T = new int* [N];
  for(int j = 0 ; j < N ; j++)
  {
    M_T[j] = new int [N];
    for(int k = 0 ; k < N ; k++)
    {
      M_T[j][k] = M[k][j];
    }
  }

  long long * oneDMat = new long long [N*N];

  for (int i = 0 ; i < N ; i++)
    for (int k = 0 ; k < N ; k++)
      oneDMat[i*N+k] = M_T[i][k];

  ////CONVERTING MATRIX TO ARRAY/////----------------------------------------

  double p = 1;
  double lastColumn, t_sum;

  double* x_pre_h = new double [N];
  for(int i = 0 ; i < N ; i++)
  {
    //lastColumn = M[i][N-1]; //--without transpose
    lastColumn = M_T[N-1][i];
    t_sum = 0;

    for(int j = 0 ; j < N ; j++)
      t_sum += M[i][j];  // M normally

    x_pre_h[i] = lastColumn - t_sum/2.0;
    p *= x_pre_h[i];
  }

  long long th_num;
  dim3 my_block,my_grid;
  //FINE TUNING
  if(tn11 == ((long long)1 << 14 ))
  {
    dim3 block(128,1,1);
    my_block = block;
    dim3 grid(128,1,1);
    my_grid = grid;
    th_num = ( (long long)1 << 14 );
  }
  else if(tn11 == ((long long)1 << 19 ))
  {
    dim3 block(512,1,1);
    my_block = block;
    dim3 grid(1024,1,1);
    my_grid = grid;
    th_num = ( (long long)1 << 19 );
  }
  else
  {
    dim3 block(1024,1,1);
    my_block = block;
    dim3 grid(1024,1,1);
    my_grid = grid;
    th_num = ( (long long)1 << 20 );
  }

  cout << "Thread num: " << th_num << endl;
  cout << "tn11: " << tn11 << endl;
  cout << "Chunk size: " << tn11 / th_num << endl;
/*
  long long* chunk_ptr;
  cudaMalloc((void**)&chunk_ptr, sizeof(long long));
  cudaMemcpy(chunk_ptr, &chunk, sizeof(long long), cudaMemcpyHostToDevice);
*/
/*
  long long* tn11_ptr;
  cudaMalloc((void**)&tn11_ptr, sizeof(long long));
  cudaMemcpy(tn11_ptr, &tn11, sizeof(long long), cudaMemcpyHostToDevice);
*/
  double* x_pre_d;
  cudaMalloc((void**)&x_pre_d, N*sizeof(double));
  cudaMemcpy(x_pre_d, x_pre_h, sizeof(double)*N, cudaMemcpyHostToDevice);

  double* x = new double [N*th_num];
  double* x_k;
  cudaMalloc((void**)&x_k, th_num*N*sizeof(double));
  //  cudaMemcpy(x_k, x, sizeof(double)*N*th_num, cudaMemcpyHostToDevice);

  double* p_ptr;
  cudaMalloc((void**)&p_ptr, sizeof(double));
  cudaMemcpy(p_ptr, &p, sizeof(double), cudaMemcpyHostToDevice);

  long long* myMat;
  cudaMalloc((void**)&myMat, sizeof(long long)*N*N);
  cudaMemcpy(myMat, oneDMat, sizeof(long long)*N*N, cudaMemcpyHostToDevice);

  //// ITERATION STARTS HERE/////-------------------------------------------
  start = omp_get_wtime();


  for (int i = 0 ; i < N ; i++) {
    for (int k = 0 ; k < N ; k++) {
      cout << oneDMat[i * N + k] << " ";
    }
    cout << endl;
  }
  
  //perMat<<32,1024>>(x, x_pre_h, N, oneDMat, p, tn11, chunk);
  perMat<<<my_grid,my_block>>>(x_k, x_pre_d, N, myMat, p_ptr, th_num);
  cudaDeviceSynchronize();
  cudaCheckError();
  cout << "1 | " << p << endl;

  cudaMemcpy(&p, p_ptr, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cout << "2 | " << p << endl;

  double result = (4 * (N & 1) - 2) * p;
  //// ITERATION ENDS HERE/////---------------------------------------------
 end = omp_get_wtime();
 cout << "Threads: " << th_num << "\tResult:" << result << "\tTime:" << end - start << " s" << endl;

  cudaFree(x_k);
  cudaFree(myMat);
  cudaFree(x_pre_d);
  cudaFree(p_ptr);

   return 0;
  }
