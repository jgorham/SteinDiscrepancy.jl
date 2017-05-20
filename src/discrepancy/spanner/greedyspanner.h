#ifndef GREEDYSPANNER_H
#define GREEDYSPANNER_H

#include "structures.h"
#include "binaryheap.h"
#include <cmath>
#include <cstdio>
#include <unordered_set>

inline double distance(const vertex2 &a, const double x, const double y)
{
  // hypot from cmath returns hypotenuse of right angle triangle
  return hypot(a.x-x,a.y-y);
}

// Computes Euclidean distance between 2-dimensional vertices
inline double distance(const vertex2 &a, const vertex2 &b)
{
  return distance(a, b.x, b.y);
}

// Computes L1 distance between vertices
inline double distance(const vertex &a, const vertex &b)
{
  double l1_distance = 0;
  for(int ii = 0; ii < a.size; ii++) {
    l1_distance += fabs(a.data[ii] - b.data[ii]);
  }
  return l1_distance;
}

template<class Heap>
void DoDijkstra(unsigned int j,
                const pointset& vertices,
                std::vector< std::pair<unsigned int, double> >* myEdges,
                Heap &myHeap,
                double t,
                std::vector<int>& QContent,
                Heap& Q,
                bool* dirtyBits,
                const std::unordered_set<edge, edge_hash>& valid_edges_hashset
                ) {
  unsigned int N = vertices.size();
  bool use_full_graph = valid_edges_hashset.empty();
  QContent[j] = -1;
  dirtyBits[j] = false;
  myHeap.clear();
  for (unsigned int i = 0; i < N; i++)
    myHeap.insert(i, std::numeric_limits<double>::infinity());
  myHeap.decreaseKey(j, 0);
  while (myHeap.getCount() > 0) {
    std::pair<unsigned int, double> pair = myHeap.getMin();
    double directDist = distance(vertices[pair.first], vertices[j]);
    // If not using full graph, make sure (j, pair.first) is a valid edge
    if (pair.second > t * directDist && (use_full_graph || valid_edges_hashset.count(edge(j, pair.first))))
    {
      /// LM: Changed this line to refer to first coordinate
      /// of a vertex of arbitrary length
      ///if (vertices[pair.first].x >= vertices[j].x) {
      if (vertices[pair.first].data[0] >= vertices[j].data[0]) {
        if (QContent[j] == -1 || directDist < distance(vertices[j], vertices[QContent[j]]))
          QContent[j] = pair.first;
      }
    }
    myHeap.extractMin();
    for (unsigned int i = 0; i < myEdges[pair.first].size(); i++) {
      std::pair<unsigned int, double> edge = myEdges[pair.first][i];
      double alt = pair.second + edge.second;
      if (alt < myHeap.getValue(edge.first))
        myHeap.decreaseKey(edge.first, alt);
    }
  }
  if (QContent[j] != -1) {
    if (!Q.contains(j))
      Q.insert(j, distance(vertices[QContent[j]], vertices[j]));
    else {
      if (Q.getValue(j) < distance(vertices[QContent[j]], vertices[j]))
        Q.increaseKey(j, distance(vertices[QContent[j]], vertices[j]));
      else
        Q.decreaseKey(j, distance(vertices[QContent[j]], vertices[j]));
    }
  }
  else if (Q.contains(j))
    Q.remove(j);
}

template<class Heap>
int GreedyLinspace3(const pointset &vertices, double t, edgelist &edges,
                    const edgelist &valid_edges_array) {
  unsigned int N = vertices.size();
  unsigned int edgeCount = 0;
  t = (t < 2.0) ? t : 2.0;
  Heap myHeap(N, 0);
  
  // Insert each row of valid_edges_array into hashset. If valid_edges_array is
  // empty, full graph is used.
  std::unordered_set<edge, edge_hash> valid_edges_hashset;
  for (edge e : valid_edges_array)
    valid_edges_hashset.insert(e);

  std::vector<std::pair<unsigned int, double> >* myEdges = new std::vector<std::pair<unsigned int, double> >[N];
  std::vector<int> QContent(N, -1);
  Heap Q(N, -std::numeric_limits<double>::infinity());
  bool* dirtyBits = new bool[N];
  for (unsigned int j = 0; j < N; j++)
    DoDijkstra<Heap>(j, vertices, myEdges, myHeap, t, QContent, Q, dirtyBits, valid_edges_hashset);

  while (Q.getCount() > 0) {
    bool foundOne = false;
    std::pair<unsigned int, double> minQ;
    while (!foundOne)
    {
      minQ = Q.getMin();
      if (dirtyBits[minQ.first])
        DoDijkstra<Heap>(minQ.first, vertices, myEdges, myHeap, t, QContent, Q, dirtyBits, valid_edges_hashset);
      else
        foundOne = true;
      if (Q.getCount() == 0)
        goto end;
    }
    int minRepFrom = minQ.first;
    int minRepTo = QContent[minQ.first];
    double minRepDist = minQ.second;

    std::pair<unsigned int, double> pair(minRepFrom, minRepDist);
    myEdges[minRepTo].push_back(pair);
    pair.first = minRepTo;
    myEdges[minRepFrom].push_back(pair);

    edges.push_back(edge(minRepFrom,minRepTo));

    edgeCount++;

    for (unsigned int j = 0; j < N; j++)
      if (QContent[j] != -1)
        dirtyBits[j] = true;
  }
end:

  delete[] dirtyBits;
  delete[] myEdges;
  return edgeCount;
}

#endif
