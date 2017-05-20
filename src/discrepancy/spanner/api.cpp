#include "assert.h"
#include <iostream>
#include <vector>
#include "greedyspanner.h"

using namespace std;

struct EdgeData {
  int num_edges;
  void* E;
};

// Finds a geometric spanner with dilation factor at most t for a given
// point set
//
// Args:
//   pointlist: array representing n x p matrix of points in row order
//   n: number of points
//   p: length of each point
//   t: spanner dilation factor
//   valid_edges: array representing num_valid_edges x 2 matrix of valid
//     edges in row order. edges must be sorted so that the lower-numbered
//     vertex is e.x and the higher-numbered vertex is e.y.
//   num_valid_edges: number of rows in valid_edges. if 0, valid_edges is ignored
//      and the complete graph is used.
extern "C" void* prune_edges(
    double* pointlist, 
    int n,
    int p,
    double t,
    int* valid_edges,
    int num_valid_edges
) {
  edgelist* E = new edgelist();
  pointset* V = new pointset();
  for (int ii = 0; ii < p*n; ii += p) {
    //V->push_back(vertex(pointlist[ii], pointlist[ii+1]));
    V->push_back(vertex(p, pointlist+ii));
  }
  
  edgelist* valid_edges_array = new edgelist();
  for (int ii = 0; ii < 2*num_valid_edges; ii += 2) {
    valid_edges_array->push_back(edge(valid_edges[ii], valid_edges[ii+1]));
  }
  
  int num_edges = GreedyLinspace3<BinHeap<double> >(*V, t, *E, *valid_edges_array);
  delete V;  // free vector of vertices
  delete valid_edges_array;
  EdgeData* ed = new EdgeData;
  ed->num_edges = num_edges;
  ed->E = (void*)E;
  return (void*)ed;
};

extern "C" void load_edges(
  void* void_ed,
  unsigned int* julia_edges
) {
  EdgeData* ed = (EdgeData*)void_ed;
  edgelist* E = (edgelist*)ed->E;

  for (unsigned int ii = 0; ii < E->size(); ii++) {
    julia_edges[2*ii] = (*E)[ii].x;
    julia_edges[2*ii+1] = (*E)[ii].y;
  }

  // free all the memory
  delete E;
  delete ed;
  return;
}
