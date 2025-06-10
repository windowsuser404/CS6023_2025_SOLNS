#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <ostream>

#define DEBUG 0
#define MOD 1000000007
#define INVALID_EDGE (UINT_MAX)

using std::cin;
using std::cout;
using uint = unsigned;

struct Edge {
  uint src, dest, weight;
  char multiplier;
};

struct Vertex {
  uint root;
  uint cheapest_edge;
};

struct Graph {
  Vertex *vertices;
  Edge *edges;
  bool *mst_edges;
  uint V_count;
  uint E_count;
};

__constant__ Graph d_graph;
__device__ uint total_unions;

void checkCudaError() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (" << err << ")"
              << std::endl;
    exit(1);
  } else {
    std::cerr << "No CUDA errors detected." << std::endl;
  }
}

__device__ inline int compare_edges(const Edge *edges, const uint &first,
                                    uint &second) {
  if (first == second)
    return 0;

  const Edge &f_edge = edges[first];
  const Edge &s_edge = edges[second];

  if (f_edge.weight < s_edge.weight)
    return -1;
  else if (f_edge.weight > s_edge.weight)
    return 1;

  if (first < second)
    return -1;
  else
    return 1;
}

__device__ inline uint find_root(const Vertex *root_list, const uint vertex) {
  return root_list[vertex].root;
}

__device__ inline void union_join(Vertex *root_list, const uint first,
                                  const uint second) {
  root_list[first].root = second;
}

__device__ inline void path_compression(Vertex *root_list, const uint i) {
  uint root_v = root_list[i].root;

  while (root_list[root_v].root != root_v) {
    root_v = root_list[root_v].root;
  }

  root_list[i].root = root_v;
}

// multiplies all needed weights in graph
__global__ void parse_graph() {
  const int bID = blockIdx.x;
  const int block_width = blockDim.x;

  const uint E_count = d_graph.E_count;
  Edge *const edges = d_graph.edges;

  // no need to max, it will atleast be 1, any number as long we cover fully is
  // fine
  const uint e_per_blk = E_count / gridDim.x + 1;

  const uint block_start = bID * e_per_blk;
  const uint block_end = min((bID + 1) * e_per_blk, E_count);

  for (uint i = block_start + threadIdx.x; i < block_end; i += block_width) {
    // edges[i].weight = edges[i].weight * edges[i].multiplier;
    int minsrc = min(edges[i].src, edges[i].dest);
    int maxdest = max(edges[i].src, edges[i].dest);
    edges[i].src = minsrc;
    edges[i].dest = maxdest;

    switch (edges[i].multiplier) {
    case 't':
      edges[i].weight *= 5;
      break;
    case 'd':
      edges[i].weight *= 3;
      break;
    case 'g':
      edges[i].weight *= 2;
      break;
    // normal edge
    default:
      break;
    }
  }
}

__global__ void fill_arrs() {
  const int bID = blockIdx.x;
  const int b_size = blockDim.x;

  const uint V_count = d_graph.V_count;
  Vertex *const vertices = d_graph.vertices;

  // no need to max, it will atleast be 1, any number as long we cover fully is
  // fine
  const uint v_per_blk = V_count / gridDim.x + 1;

  const uint b_start = bID * v_per_blk;
  const uint b_end = min((bID + 1) * v_per_blk, V_count);

#if DEBUG
  printf("B_strt = %d, B_end = %d by tid = %d, bid = %d, v_per_blk = %d\n",
         b_start, b_end, threadIdx.x, blockIdx.x, v_per_blk);
#endif

  for (uint i = b_start + threadIdx.x; i < b_end; i += b_size) {
    vertices[i] = Vertex{i, INVALID_EDGE};
  }
}

__global__ void reset_all_arr() {
  const int bID = blockIdx.x;
  const int b_width = blockDim.x;

  const uint V_count = d_graph.V_count;
  Vertex *const vertices = d_graph.vertices;

  // no need to max, it will atleast be 1, any number as long we cover fully is
  // fine
  const uint v_per_blk = V_count / gridDim.x + 1;

  const uint b_start = bID * v_per_blk;
  const uint b_end = min((bID + 1) * v_per_blk, V_count);

  for (uint i = b_start + threadIdx.x; i < b_end; i += b_width) {
    vertices[i].cheapest_edge = INVALID_EDGE;
    path_compression(vertices, i);
  }
}

__global__ void min_edges() {
  const int bID = blockIdx.x;
  const int block_width = blockDim.x;

  const uint E_count = d_graph.E_count;
  Vertex *const vertices = d_graph.vertices;
  Edge *const edges = d_graph.edges;

  // no need to max, it will atleast be 1, any number as long we cover fully is
  // fine
  const uint e_per_blk = E_count / gridDim.x + 1;

  const uint block_start = bID * e_per_blk;
  const uint block_end = min((bID + 1) * e_per_blk, E_count);

  for (uint i = block_start + threadIdx.x; i < block_end; i += block_width) {
    __syncwarp();
    Edge &e = edges[i];
    e.src = find_root(vertices, e.src);
    e.dest = find_root(vertices, e.dest);

    if (e.src == e.dest) {
      continue;
    }

    // Atomic update cheapest_edge
    uint expected = vertices[e.src].cheapest_edge;
    uint old;

    // copying from cuda example
    while (expected == INVALID_EDGE || compare_edges(edges, i, expected) < 0) {
      old = atomicCAS(&vertices[e.src].cheapest_edge, expected, i);
      if (expected == old) {
        break;
      }
      expected = old;
    }

    expected = vertices[e.dest].cheapest_edge;

    // copying from cuda example
    while (expected == INVALID_EDGE || compare_edges(edges, i, expected) < 0) {
      old = atomicCAS(&vertices[e.dest].cheapest_edge, expected, i);
      if (expected == old) {
        break;
      }
      expected = old;
    }
  }
}

__global__ void join_sub_trees() {
  const int bID = blockIdx.x;
  const int b_width = blockDim.x;

  const uint V_count = d_graph.V_count;
  Vertex *const vertices = d_graph.vertices;
  Edge *const edges = d_graph.edges;

  // no need to max, it will atleast be 1, any number as long we cover fully is
  // fine
  const uint v_per_blk = V_count / gridDim.x + 1;

  const uint b_start = bID * v_per_blk;
  const uint b_end = min((bID + 1) * v_per_blk, V_count);

  uint n_unions_made = 0;
  for (uint i = b_start + threadIdx.x; i < b_end; i += b_width) {
    const uint edge_ind = vertices[i].cheapest_edge;

    if (edge_ind == INVALID_EDGE) {
      continue;
    }

    const Edge &edge_ptr = edges[edge_ind];

    if (edge_ptr.dest == i &&
        edge_ind == vertices[edge_ptr.src].cheapest_edge) {
      continue;
    }

    const uint j =
        (i == edge_ptr.src ? edge_ptr.dest
                           : edge_ptr.src); // this is the other index
#if DEBUG
    printf("%d edge in, by tid = %d bid = %d\n", edge_ind, threadIdx.x, bID);
#endif

    d_graph.mst_edges[edge_ind] = 1;

#if DEBUG
    printf("Joining %d %d\n", i, j);
#endif

    union_join(vertices, i, j);
    n_unions_made++;
  }

  atomicAdd(&total_unions, n_unions_made);
}

__global__ void calc_tree_wt(uint *d_mst_wt) {
  const int bID = blockIdx.x;
  const int b_width = blockDim.x;

  const uint E_count = d_graph.E_count;

  // no need to max, it will atleast be 1, any number as long we cover fully is
  // fine
  const uint e_per_blk = E_count / gridDim.x + 1;

  const uint b_start = bID * e_per_blk;
  const uint b_end = min((bID + 1) * e_per_blk, E_count);

  uint wt = 0;
  for (uint i = b_start + threadIdx.x; i < b_end; i += b_width) {
    if (d_graph.mst_edges[i]) {
      wt += d_graph.edges[i].weight;
    }
  }
  atomicAdd(d_mst_wt, wt);
}

int main() {
  uint V;
  cin >> V;
  uint E;
  cin >> E;
  Edge *edges = new Edge[E];
  Graph graph;
  uint mst_wt;

  for (uint i = 0; i < E; i++) {
    cin >> edges[i].src;
    cin >> edges[i].dest;
    cin >> edges[i].weight;
    cin >> edges[i].multiplier;
    cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  // device vars
  bool *d_mst_edges;
  Vertex *d_vertices;
  Edge *d_edges;
  uint *d_mst_wt;

  cudaMalloc(&d_mst_edges, sizeof(bool) * E);
  cudaMemset(d_mst_edges, 0, sizeof(bool) * E);

  cudaMalloc(&d_vertices, sizeof(Vertex) * V);

  cudaMalloc(&d_mst_wt, sizeof(uint));
  cudaMemset(d_mst_wt, 0, sizeof(uint));

  cudaMalloc(&d_edges, sizeof(Edge) * E);
  cudaMemcpy(d_edges, edges, sizeof(Edge) * E, cudaMemcpyHostToDevice);

  graph.edges = d_edges;
  graph.E_count = E;
  graph.V_count = V;
  graph.vertices = d_vertices;
  graph.mst_edges = d_mst_edges;

  cudaMemcpyToSymbol(d_graph, &graph, sizeof(Graph));

  // use unionold to figure out minimum spanning forests is formed
  uint sub_trees = 0;
  uint sub_trees_prev = 0;

  // initialise all block and threads
  uint t_per_blk = 1024;

#if DEBUG
  printf("Doing %d Vertices with %d Edges\n", V, E);
  printf("With per_block %d threads\n", t_per_blk);
  printf("%d blocks for edges\n", edges_per_blk);
  printf("%d blocks for vertices\n", verts_per_blk);
#endif

  dim3 block_size(t_per_blk, 1, 1);

  cudaMemcpyToSymbol(total_unions, &sub_trees, sizeof(uint));

#if DEBUG
  checkCudaError();
#endif

  // Answer should be calculated in Kernel. No operations should be performed
  // here. Only copy data to device, kernel call, copy data back to host, and
  // print the answer.
  auto start = std::chrono::high_resolution_clock::now();
  // Kernel call(s) here

  // doing division inside timing, no computation in cpu is ignored
  uint edges_per_blk = (uint)(ceil(E / (float)t_per_blk));
  uint verts_per_blk = (uint)(ceil(V / (float)t_per_blk));
  dim3 gridE_size(128, 1, 1);
  dim3 gridV_size(verts_per_blk, 1, 1);

#if DEBUG
  uint iter = 0;
#endif

  parse_graph<<<gridE_size, block_size>>>();
  fill_arrs<<<gridV_size, block_size>>>();
  do {

#if DEBUG
    printf("Doing iteration %d with %d sub_trees\n", iter++, sub_trees);
#endif

    sub_trees_prev = sub_trees;

    reset_all_arr<<<gridV_size, block_size>>>();
    min_edges<<<gridE_size, block_size>>>();
    join_sub_trees<<<gridV_size, block_size>>>();
    cudaMemcpyFromSymbol(&sub_trees, total_unions, sizeof(uint));

#if DEBUG
    checkCudaError();
#endif

  } while (sub_trees != sub_trees_prev && sub_trees < V - 1);

  calc_tree_wt<<<gridE_size, block_size>>>(d_mst_wt);

#if DEBUG
  checkCudaError();
#endif

  auto end = std::chrono::high_resolution_clock::now();
  cudaMemcpy(&mst_wt, d_mst_wt, sizeof(uint), cudaMemcpyDeviceToHost);
  std::chrono::duration<double> elapsed1 = end - start;
  // Print only the total MST weight
  cout << mst_wt % MOD << std::endl;

  // cout << elapsed1.count() << " s\n";
  return 0;
}
