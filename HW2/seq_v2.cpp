#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
#include <string>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <boost/functional/hash.hpp>

using namespace std;
struct CRS
{
  int* col;
  int* rowptr;
  int rowsize;
  int nonzerosize;

  CRS(int numRows, int numNZ){
    col = new int[numNZ];
    rowptr = new int[numRows+1];
    rowsize = numRows;
    nonzerosize = numNZ;
  }
};

bool compareData(pair<int, int> A,  pair<int, int>  B)
{
  return (A.second < B.second) || ( (A.second == B.second) && (A.first<B.first ) );
}

void printCRS(CRS & storage)
{
  cout<<"col values are "<<endl;
  for(int i = 0; i <storage.nonzerosize; i++)
    cout<<storage.col[i]<<" ";
  cout<<endl;
  cout<<"rowptr values are "<<endl;
  for(int i = 0; i <= storage.rowsize; i++)
    cout<<storage.rowptr[i]<<" ";
  cout<<endl;
}

void usage()
{
  cout << "USAGE: ./exec <filename>" << endl;
  exit(0);
}

int findMaxColor (int* & graphColor, int numRows)
{
  int max = -1;
  for (int i = 0 ; i < numRows ; i++)
  {
    //cout << graphColor[i] << endl;
    if (graphColor[i] > max)
      max = graphColor[i];
  }
  return max+1;
}

bool checkValidity(CRS & storage, int* & graphColor)
{
  int start,end;
  for (int i = 0 ; i < storage.rowsize ; i++)
  {
    start = storage.rowptr[i];
    end = storage.rowptr[i+1];
    for (start ; start < end ; start++)
    {
      if(graphColor[storage.col[start]] == graphColor[i])
      {
        cout << "Collision Spotted !" << endl;
        cout << "Collision between: " << i << " " << storage.col[start] << endl;
        cout << "Colors are " << graphColor[i] << " " << graphColor[storage.col[start]] << endl;
	// return false;
      }
    }
  }
  cout << "No Collisions !" << endl;
  return true;
}

void seqColor_v2(CRS & storage, int* & graphColor, int* available)
{
  int i,end;
  //cout << storage.rowsize << " " << storage.nonzerosize << endl;
  //printCRS(storage);
  for(int v = 0; v < storage.rowsize; v++)
  {
    i = storage.rowptr[v];
    end = storage.rowptr[v+1];
    //cout << v << " " << i << " " << end << endl;
    for(i ; i < end ; i++)
    {
      if (graphColor[storage.col[i]] != -1)
        available[graphColor[storage.col[i]]] = v;
    }
    for(int c = 0; c < storage.rowsize; c++)
    {
      if(available[c] != v)
      {
        //cout << v << " " << c << endl;
        graphColor[v] = c;
        break;
      }
    }
  }
}

void parColor (CRS & storage, int* & graphColor,int* & available, int* & reColor)
{
  for(int t = 1; t <=16; t*=2)
  {
    int ct = storage.rowsize;
    int a = 0;
    double s1 = omp_get_wtime();
    while(ct > 0)
    {
      cout << ct << endl;
      #pragma omp parallel proc_bind(spread) num_threads(t) shared(ct)
      {
        int* ava = new int [storage.rowsize];
        for (int j = 0 ; j < storage.rowsize ; j++) ava[j] = available[j];

        #pragma omp for schedule(guided)
        for (int v = 0 ; v < ct ; v++)
        {
	  int curr_vertex = reColor[v];
          int start = storage.rowptr[curr_vertex];
          int end = storage.rowptr[curr_vertex + 1];
          for (start ; start < end ; start++) {
            ava[graphColor[storage.col[start]]] = curr_vertex;
	  }
     
          for (int c = 0 ; c < storage.rowsize ; c++) {
            if (ava[c] != curr_vertex) {
              graphColor[curr_vertex] = c;
              break;
            }
	  }
        }
      }

      a = 0;
      #pragma omp parallel proc_bind(spread) num_threads(t) shared(ct,a)
      {
        #pragma omp for schedule(guided) //reduction(+:a)
        for (int i = 0 ; i < ct ; i++)
        {
	  int curr_vertex = reColor[i];
          int start = storage.rowptr[curr_vertex];
          int end = storage.rowptr[curr_vertex+1];
          for (start ; start < end ; start++)
          {
	    int nbr_vertex = storage.col[start];
            if ((graphColor[nbr_vertex] == graphColor[curr_vertex]) && (curr_vertex < nbr_vertex))
            {
              #pragma omp critical
              {
                //cout << a << endl;
                reColor[a] = curr_vertex;
                //cout << reColor[a] << endl;
                a += 1;
              }
              //cout << graphColor[storage.col[start]] << " " << graphColor[i] << " " << a << endl;
            }
          }
        }
      }
      cout << "Number of vertices that needs to be re-colored: " << a << endl;
      ct = a;
    }
    double s2 = omp_get_wtime();
    cout << "Total time for " << t << " thread(s) are: " << s2 - s1 << endl;
    cout << "Max color is " << findMaxColor(graphColor, storage.rowsize) << endl;
    checkValidity(storage, graphColor);
    //for (int i = 0 ; i < storage.rowsize ; i++) cout << graphColor[i] << " ";
    //cout << endl;
    for (int j = 0 ; j < storage.rowsize ; j++) graphColor[j] = -1;
    //for (int j = 0 ; j < storage.rowsize ; j++) available[j] = -10;
    for (int j = 0 ; j < storage.rowsize ; j++) reColor[j] = j;
  }
}

void seqColor(CRS & storage, int* graphColor, int* available)
{
  graphColor[0] = 0;
  for (int i = 0 ; i < storage.rowsize ; i++)
  {
    for (int k = storage.rowptr[i] ; k != storage.rowptr[i+1] ; k++)
    {
      if (graphColor[storage.col[k]] != -1)
      {
        available[graphColor[storage.col[k]]] = 1;
      }
    }
    int ct;
    for (ct = 0; ct < storage.rowsize ; ct++)
    {
      if (available[ct] == 0)
      {
        break;
      }
    }
    graphColor[i] = ct;

    for (int k = storage.rowptr[i] ; k != storage.rowptr[i+1] ; k++)
    {
      if (graphColor[storage.col[k]] != -1)
      {
        available[graphColor[storage.col[k]]] = 0;
      }
    }
  }
}

int main(int argc, const char** argv)
{
  if(argc != 2)
    usage();
  double start = omp_get_wtime();
  unsigned int numRows, numCols, numNZ, edgeStart, edgeEnd, count;
  string line;
  string patternType = "";
  const char* filename = argv[1];
  unsigned int realNumNZ;

  ifstream input (filename);

  if(input.fail())
    return 0;
  getline(input, line);
  stringstream ss(line);
  string temp;
  while(ss>>temp);
  patternType = temp;

  while(getline(input, line))
  {
    stringstream ss(line);
    ss>>temp;
    if(temp.substr(0,1) != "%")
    {
      numRows = stoul(temp);
      ss>>numCols>>numNZ;
      break;
    }
  }

  cout<<numRows<<" "<<numCols<<" "<<numNZ<<endl;
  int counter = 0;
  typedef pair<int, int> nodePair;
  vector<nodePair> edges(2*numNZ,{0, 0});
  while(getline(input, line)){
    int row, col;
    stringstream ss(line);
    ss>>col>>row;

    edges[counter] = make_pair(col, row);
    edges[numNZ+counter] = make_pair(row, col);
    counter++;
  }
  sort(edges.begin(), edges.end(), compareData);

  auto last = unique(edges.begin(), edges.end());
  edges.erase(last, edges.end());
  edges.erase(remove_if(
			 edges.begin(),edges.end(),
			 [](const nodePair& p) {
			   return p.first == p.second; // put your condition here
			 }), edges.end());

  if(edges[0].second != 0)
  for(int i = 0; i<edges.size(); i++)
  {
    edges[i].first--;
    edges[i].second--;
  }
  /*
  for(int i = 0; i<edges.size(); i++)
  {
    if(edges[i].first == edges[i].second)
      cout<<"edge"<<edges[i].first<<" "<<edges[i].second<<endl;
  }
  */
  realNumNZ = edges.size();

  CRS storage(numRows, realNumNZ);

  //storage.col[0] = edges[0].first;
  //storage.rowptr[0] = edges[0].second;
  unsigned int prevRow = edges[0].second;

  for(int i = 0; i<realNumNZ; i++)
  {
    storage.col[i] = edges[i].first;
    if(edges[i].second != prevRow)
    {
      prevRow = edges[i].second;
      storage.rowptr[edges[i].second] = i;
    }
  }

  storage.rowptr[numRows] = realNumNZ;

  cout<<"Reading file lasted "<<omp_get_wtime() - start<<" seconds"<<endl;
  cout<<"realnumnz is "<<edges.size()<<endl;

  //printCRS(storage);

  int* graphColor = new int [numRows];
  for (int j = 0 ; j < numRows ; j++) graphColor[j] = -1;
// -1
  int* available = new int [numRows];
  for (int j = 0 ; j < numRows ; j++) available[j] = -10;
// -1
  int* reColor = new int [numRows];
  for (int j = 0 ; j < numRows ; j++) reColor[j] = j;

  double s1 = omp_get_wtime();
  //seqColor(storage, graphColor, available);
  //seqColor_v2(storage, graphColor, available);
  parColor(storage, graphColor, available, reColor);
  double s2 = omp_get_wtime();
  /*
  cout << "Total Time: " << s2 - s1 << endl;
  cout << "Max Color: " << findMaxColor(graphColor, numRows) << endl;
  checkValidity(storage, graphColor);
  */
  delete [] graphColor;
  delete [] available;
  delete [] reColor;
  return 0;
}

/*
void parColor (CRS & storage, int* & graphColor, int* & available, bool* & reColor)
{
  cout << 0 << endl;
  for(int t = 1; t <=16; t*=2)
  {
    bool willContinue = true;
    double s1 = omp_get_wtime();
    cout << 1 << endl;
    while (willContinue)
    {
      int ct;
      #pragma omp parallel num_threads(t) proc_bind(spread)
      {
        cout << 2 << endl;
        //int tid = omp_get_thread_num();
        //int rSize = storage.rowsize;
        //int startId = (tid * rSize)/t;
        #pragma omp for schedule(static)
          for (int v = 0 ; v < storage.rowsize ; v++)
          {
            if (reColor[v] == true)
            {
              int start = storage.rowptr[v];
              int end = storage.rowptr[v+1];

              for (start ; start < end ; start++)
                if (graphColor[storage.col[start]] != -1)
                  available[graphColor[storage.col[start]]] = v;

              for (int c = 0 ; c < storage.rowsize ; c++)
                if (available[c] != v)
                {
                  graphColor[v] = c;
                  reColor[v] = false;
                  break;
                }
            }
          }
        cout << 3 << endl;
        ct = 0;
        #pragma omp for schedule(static)
          for (int i = 0 ; i < storage.rowsize ; i++)
          {
            for (int k = storage.rowptr[i] ; k < storage.rowptr[i+1] ; k++)
            {
              if(graphColor[storage.col[k]] == graphColor[i] && i > storage.col[k])
              {
                reColor[i] = true;
                ct += 1;
              }
            }
          }
      }
      cout << 4 << endl;
      if (ct > 0)
        willContinue = false;
    }
    double s2 = omp_get_wtime();
    cout << 5 << endl;

    cout << "Total time is: " << s2 - s1 << endl;
    cout << "Max color is: " << findMaxColor(graphColor, storage.rowsize) << endl;

    //PREPROCESSING PART AGAIN
    for (int j = 0 ; j < storage.rowsize ; j++) graphColor[j] = -1;
    for (int j = 0 ; j < storage.rowsize ; j++) available[j] = -10;
    for (int j = 0 ; j < storage.rowsize ; j++) reColor[j] = true;

    cout << 6 << endl;
  }
}
*/

/*
void parColor (CRS & storage, int* & graphColor, int* & available, bool* & reColor)
{
  for(int t = 1; t <=16; t*=2)
  {
    bool willContinue = true;
    double s1 = omp_get_wtime();
    while (willContinue)
    {
      int ct;
      #pragma omp parallel num_threads(t) proc_bind(spread)
      {
        int tid = omp_get_thread_num();
        int rSize = storage.rowsize;
        int startId = (tid * rSize)/t;
        int endId = ((tid+1) * rSize)/t;
        #pragma omp for schedule(static)
          for (int v = startId ; v < endId ; v++)
          {
            if (reColor[v] == true)
            {
              int start = storage.rowptr[v];
              int end = storage.rowptr[v+1];

              for (start ; start < end ; start++)
                if (graphColor[storage.col[start]] != -1)
                  available[graphColor[storage.col[start]]] = v;

              for (int c = 0 ; c < storage.rowsize ; c++)
                if (available[c] != v)
                {
                  graphColor[v] = c;
                  reColor[v] = false;
                  break;
                }
            }
          }
        ct = 0;
        #pragma omp for schedule(static)
          for (int i = startId ; i < endId ; i++)
          {
            for (int k = storage.rowptr[i] ; k < storage.rowptr[i+1] ; k++)
            {
              if(graphColor[storage.col[k]] == graphColor[i] && i > storage.col[k])
              {
                reColor[i] = true;
                ct += 1;
              }
            }
          }
        if (ct == 0)
          willContinue = false;
      }
    double s2 = omp_get_wtime();
    cout << "Total time is: " << s2 - s1 << endl;
    cout << "Max color is: " << findMaxColor(graphColor, storage.rowsize) << endl;
    }
    checkValidity(storage, graphColor);
    //PREPROCESSING PART AGAIN
    for (int j = 0 ; j < storage.rowsize ; j++) graphColor[j] = -1;
    for (int j = 0 ; j < storage.rowsize ; j++) available[j] = -10;
    for (int j = 0 ; j < storage.rowsize ; j++) reColor[j] = true;
  }
}
*/
/*
void parColor (CRS & storage, int* & graphColor,int* & available, int* & reColor)
{
  for(int t = 1; t <=16; t*=2)
  {
    int ct = storage.rowsize;
    int a = 0;
    double s1 = omp_get_wtime();
    while(ct != 0)
    {
      cout << ct << endl;
      #pragma omp parallel proc_bind(spread) num_threads(t) shared(ct)
      {
        int* ava = new int [storage.rowsize];
        for (int j = 0 ; j < storage.rowsize ; j++) ava[j] = available[j];

        #pragma omp for schedule(guided)
        for (int v = 0 ; v < ct ; v++)
        {
          int start = storage.rowptr[reColor[v]];
          int end = storage.rowptr[reColor[v]+1];
          for (start ; start < end ; start++)
            ava[graphColor[storage.col[start]]] = v;
            //if (graphColor[storage.col[start]] != -1)// -- PROBLEM HERE

          for (int c = 0 ; c < storage.rowsize ; c++)
            if (ava[c] != v)
            {
              graphColor[v] = c;
              break;
            }
        }
      }
      #pragma omp parallel proc_bind(spread) num_threads(t) shared(ct,a)
      {
        #pragma omp for schedule(guided) //reduction(+:a)
        for (int i = 0 ; i < ct ; i++)
        {
          int start = storage.rowptr[reColor[i]];
          int end = storage.rowptr[reColor[i]+1];
          for (start ; start < end ; start++)
          {
            if ((graphColor[storage.col[start]] == graphColor[i]) && (i > storage.col[start]))
            {
              #pragma omp critical
              {
                //cout << a << endl;
                reColor[a] = i;
                //cout << reColor[a] << endl;
                a += 1;
              }
              //cout << graphColor[storage.col[start]] << " " << graphColor[i] << " " << a << endl;
            }
          }
        }
      }
      cout << "Number of vertices that needs to be re-colored: " << a << endl;
      ct = a;
      a = 0;
    }
    double s2 = omp_get_wtime();
    cout << "Total time for " << t << " thread(s) are: " << s2 - s1 << endl;
    cout << "Max color is " << findMaxColor(graphColor, storage.rowsize) << endl;
    //checkValidity(storage, graphColor);
    //for (int i = 0 ; i < storage.rowsize ; i++) cout << graphColor[i] << " ";
    //cout << endl;
    for (int j = 0 ; j < storage.rowsize ; j++) graphColor[j] = -1;
    //for (int j = 0 ; j < storage.rowsize ; j++) available[j] = -10;
    for (int j = 0 ; j < storage.rowsize ; j++) reColor[j] = j;
  }
}
*/
/*
void parColor_v2 (CRS & storage, int* & graphColor,int* & available, int* & reColor, int t)
{
  int ct = storage.rowsize;
  int a = 0;
  while(ct != 0)
  {
    #pragma omp parallel proc_bind(spread) num_threads(t) shared(ct)
    {
      int* ava = new int [storage.rowsize];
      for (int j = 0 ; j < storage.rowsize ; j++) ava[j] = available[j];
      #pragma omp for schedule(guided)
      for (int v = 0 ; v < ct ; v++)
      {
        int start = storage.rowptr[reColor[v]];
        int end = storage.rowptr[reColor[v]+1];
        for (start ; start < end ; start++)
          ava[graphColor[storage.col[start]]] = v;
            //if (graphColor[storage.col[start]] != -1)// -- PROBLEM HERE

        for (int c = 0 ; c < storage.rowsize ; c++)
          if (ava[c] != v)
          {
            graphColor[v] = c;
            break;
          }
      }
    }
    #pragma omp parallel proc_bind(spread) num_threads(t) shared(ct,a)
    {
      #pragma omp for schedule(guided) //reduction(+:a)
      for (int i = 0 ; i < ct ; i++)
      {
        int start = storage.rowptr[reColor[i]];
        int end = storage.rowptr[reColor[i]+1];
        for (start ; start < end ; start++)
        {
          if ((graphColor[storage.col[start]] == graphColor[i]) && (i > storage.col[start]))
          {
            #pragma omp critical
            {
                //cout << a << endl;
              reColor[a] = i;
                //cout << reColor[a] << endl;
              a += 1;
            }
              //cout << graphColor[storage.col[start]] << " " << graphColor[i] << " " << a << endl;
          }
        }
      }
    }
    cout << "Number of vertices that needs to be re-colored: " << a << endl;
    ct = a;
    a = 0;
  }
}

void seqColor_v2(CRS & storage, int* & graphColor, int* available)
{
  int i,end;
  //cout << storage.rowsize << " " << storage.nonzerosize << endl;
  //printCRS(storage);
  for(int v = 0; v < storage.rowsize; v++)
  {
    i = storage.rowptr[v];
    end = storage.rowptr[v+1];
    //cout << v << " " << i << " " << end << endl;
    for(i ; i < end ; i++)
    {
      if (graphColor[storage.col[i]] != -1)
        available[graphColor[storage.col[i]]] = v;
    }
    for(int c = 0; c < storage.rowsize; c++)
    {
      if(available[c] != v)
      {
        //cout << v << " " << c << endl;
        graphColor[v] = c;
        break;
      }
    }
  }
}
*/
