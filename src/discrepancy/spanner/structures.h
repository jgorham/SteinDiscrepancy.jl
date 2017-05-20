#ifndef STRUCTURES_H
#define STRUCTURES_H

#include<vector>

// Two-dimensional vertex
struct vertex2 {
  double x,y;
 vertex2() : x(0), y(0) {}
 vertex2(double x, double y) : x(x), y(y) {}
  bool operator==(const vertex2 &v) const {return (v.x==x && v.y==y);}
};

// Vertex of arbitrary length
struct vertex {
  // Number of vertex components
  int size;
  // Pointer to vertex data
  double* data;
  vertex(int size, double* data) : size(size), data(data) {}
};

struct edge {
  unsigned int x,y;
 edge() : x(0), y(0) {}
 edge(unsigned int x, unsigned int y) : x(x), y(y) {}
  bool operator==(const edge &e) const {return (e.x==x && e.y==y) || (e.x==y && e.y==x);}
  inline bool operator < (const edge &o) const {
    if (x != o.x) return x < o.x;
    else return y < o.y;
  }
};

typedef std::vector<vertex> pointset;
typedef std::vector<edge> edgelist;

// Copied from the boost implementation of hash_combine prior to version 1.56.0
void hash_combine(std::size_t &seed, const unsigned int& x) {
  seed ^= std::hash<unsigned int>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hash function for edges
struct edge_hash {
  std::size_t operator () (const edge& e) const {
    std::size_t seed = 0;
    hash_combine(seed, std::hash<unsigned int>()(e.x));
    hash_combine(seed, std::hash<unsigned int>()(e.y));
    return seed;
  }
};

#endif
