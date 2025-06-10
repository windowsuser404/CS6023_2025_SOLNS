#include <cuda.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>

#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <limits.h>

// Macro to wrap CUDA API calls and check for errors
#define CUDA_CHECK(err)                                                        \
  if (err != cudaSuccess) {                                                    \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (code " << err \
              << "), at " << __FILE__ << ":" << __LINE__ << std::endl;         \
    exit(EXIT_FAILURE);                                                        \
  }

#define G_MAKING 0
#define CSR_DEBUG 0
#define CSR_D_DEBUG 0
#define SSSP_MAKING_DEBUG 0
#define SSSP_DEBUG 0
#define WALKING_DEBUG 0
#define DEBUG 0
#define BLOCK_SIZE 256

using namespace std;
using uint = unsigned;
using ll = long long int;

typedef struct pop_city {
  ll *id;
  ll *curr_city;
  ll *curr_road;
  ll *young;
  ll *old;
  ll *dist_walked;
} pop_city;

typedef struct Graph {
  ll vertex_count;
  ll edge_count;
  // array to see if shelter
  bool *is_shelter;
  ll *shelter_cap;
  // all destination vertices and their length
  ll *dsts;
  // we shall store source also, so that can leverage both edge list and csr
  ll *srcs;
  ll *length;
  ll *capacity;
  // to get to know range of edges for each vertex
  ll *offsets;
  int num_pop_cits;
  pop_city walk_stat;
} Graph;

// Structure to store the result
typedef struct PathResult {
  ll nearest_shelter_id; // ID of nearest shelter
  ll distance;           // Distance to the nearest shelter
  ll *path;              // Array to store the path vertices
  ll path_length;        // Number of vertices in the path
} PathResult;

__device__ Graph d_graph;
__device__ unsigned int bellman_flag;
__constant__ ll elder_max_dist;
__constant__ ll INF = LLONG_MAX / 2;
ll H_INF = LLONG_MAX / 2;

Graph h_graph;
// for clearing memory later from host side
Graph d_graph_copy;
////////////////////////////////////////////////////////////////////

/////////////////////////SSSP FUNCS/////////////////////////////////

__device__ bool compare(ll &old_dist, ll &new_dist, ll &old_src, ll &new_src) {
  if (new_dist > old_dist)
    return false;
  // else if (old_dist >= new_dist)
  //   return true;
  else {
    return true;
  }

  // old_dist == new_dist
  // no reason, priotise smaller nodes
  // if (new_src < old_src)
  //   return true;
  // else
  //   return false;
}

// CUDA kernel for Bellman-Ford edge relaxation
__global__ void bellmanFordRelax(ll *d_distances, ll *d_predecessors,
                                 bool *d_updated) {

  int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (edge_idx < d_graph.edge_count) {
    ll src_vertex = d_graph.srcs[edge_idx];

    if (src_vertex != -1) {
      ll dst_vertex = d_graph.dsts[edge_idx];
      ll edge_weight = d_graph.length[edge_idx];

      // Only relax edges where source distance is not infinity

#if SSSP_MAKING_DEBUG
      printf("doing %lld->%lld\n", src_vertex, dst_vertex);
#endif

      if (d_distances[src_vertex] != INF) {

        ll new_dist = d_distances[src_vertex] + edge_weight;
        do {
          ll old_pred = d_predecessors[dst_vertex];
          ll old_dist =
              atomicMin((unsigned long long *)&d_distances[dst_vertex],
                        (unsigned long long)new_dist);

#if SSSP_MAKING_DEBUG
          printf("old min is %lld, mine is %lld, src is %lld, dst is %lld\n",
                 old_dist, new_dist, src_vertex, dst_vertex);
#endif

          if (compare(old_dist, new_dist, old_pred, src_vertex)) {

#if SSSP_MAKING_DEBUG
            printf("%lld is preceded by %lld at distance %lld\n", dst_vertex,
                   src_vertex, new_dist);
#endif

            atomicCAS((unsigned long long *)&d_predecessors[dst_vertex],
                      (unsigned long long)old_pred,
                      (unsigned long long)src_vertex);
            *d_updated = true;
          } else {
            break;
          }
        } while (d_predecessors[dst_vertex] != src_vertex);
      }
    }
  }
}

// CUDA kernel to find nearest shelter
__global__ void findNearestShelter(ll *d_distances, ll source_city,
                                   ll *d_nearest_shelter, ll *d_min_distance) {
  ll idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < d_graph.vertex_count && d_graph.is_shelter[idx] &&
      d_graph.shelter_cap[idx] > 0) {
    do {
      ll old_near = *d_nearest_shelter;
      ll old_min = atomicMin((unsigned long long *)d_min_distance,
                             (unsigned long long)d_distances[idx]);
#if SSSP_MAKING_DEBUG
      printf("old min is %lld, mine is %lld, id is %lld\n", old_min,
             d_distances[idx], idx);
#endif

      if (compare(old_min, d_distances[idx], old_near, idx)) {

#if SSSP_MAKING_DEBUG
        printf("%lld is being pathed by id%lld\n", source_city, idx);
#endif

        atomicCAS((unsigned long long *)d_nearest_shelter,
                  (unsigned long long)old_near, (unsigned long long)idx);
      } else {
        break;
      }
    } while (*d_nearest_shelter != idx);
  }
}

// Host function to find the nearest shelter path for a single city
PathResult findNearestShelterPath(ll city_id) {
  PathResult result;
  result.nearest_shelter_id = -1;
  result.distance = H_INF;
  result.path = NULL;
  result.path_length = 0;

  // Device memory allocations
  ll *d_distances = NULL;
  ll *d_predecessors = NULL;
  bool *d_updated = NULL;
  ll *d_nearest_shelter = NULL;
  ll *d_min_distance = NULL;

  CUDA_CHECK(
      cudaMalloc((void **)&d_distances, h_graph.vertex_count * sizeof(ll)));
  CUDA_CHECK(
      cudaMalloc((void **)&d_predecessors, h_graph.vertex_count * sizeof(ll)));
  CUDA_CHECK(cudaMalloc((void **)&d_updated, sizeof(bool)));
  CUDA_CHECK(cudaMalloc((void **)&d_nearest_shelter, sizeof(ll)));
  CUDA_CHECK(cudaMalloc((void **)&d_min_distance, sizeof(ll)));

  // Initialize distances and predecessors
  ll *h_distances = (ll *)malloc(h_graph.vertex_count * sizeof(ll));
  ll *h_predecessors = (ll *)malloc(h_graph.vertex_count * sizeof(ll));

  for (ll i = 0; i < h_graph.vertex_count; i++) {
    h_distances[i] = (i == city_id) ? 0 : H_INF;
    h_predecessors[i] = -1;
  }

  CUDA_CHECK(cudaMemcpy(d_distances, h_distances,
                        h_graph.vertex_count * sizeof(ll),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_predecessors, h_predecessors,
                        h_graph.vertex_count * sizeof(ll),
                        cudaMemcpyHostToDevice));

  // Set initial values for nearest shelter and min distance
  ll init_nearest = -1;
  ll init_min_dist = H_INF;
  CUDA_CHECK(cudaMemcpy(d_nearest_shelter, &init_nearest, sizeof(ll),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_min_distance, &init_min_dist, sizeof(ll),
                        cudaMemcpyHostToDevice));

  // Define grid and block dimensions for edge operations
  int blockSize = BLOCK_SIZE;
  int gridSize = (h_graph.edge_count + blockSize - 1) / blockSize;

  // Run parallel Bellman-Ford
  for (ll i = 0; i < h_graph.vertex_count - 1; i++) {
    bool h_updated = false;
    CUDA_CHECK(cudaMemcpy(d_updated, &h_updated, sizeof(bool),
                          cudaMemcpyHostToDevice));

#if SSSP_MAKING_DEBUG
    printf("\nBellmaning %lld\n\n", city_id);
#endif
    bellmanFordRelax<<<gridSize, blockSize>>>(d_distances, d_predecessors,
                                              d_updated);
    CUDA_CHECK(cudaGetLastError());
#if SSSP_MAKING_DEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // Check if any updates occurred
    CUDA_CHECK(cudaMemcpy(&h_updated, d_updated, sizeof(bool),
                          cudaMemcpyDeviceToHost));
    if (!h_updated) {
      break; // Early termination if no updates
    }
  }

  // Define grid and block dimensions for vertex operations
  gridSize = (h_graph.vertex_count + blockSize - 1) / blockSize;

#if SSSP_MAKING_DEBUG
  printf("\nFind Shelter for %lld\n\n", city_id);
#endif
  // Find nearest shelter
  findNearestShelter<<<gridSize, blockSize>>>(
      d_distances, city_id, d_nearest_shelter, d_min_distance);
  CUDA_CHECK(cudaGetLastError());

#if SSSP_MAKING_DEBUG
  CUDA_CHECK(cudaDeviceSynchronize());
#endif

  // Copy results back to host
  CUDA_CHECK(cudaMemcpy(&result.nearest_shelter_id, d_nearest_shelter,
                        sizeof(ll), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&result.distance, d_min_distance, sizeof(ll),
                        cudaMemcpyDeviceToHost));

  // Only reconstruct path if a shelter was found
  if (result.nearest_shelter_id != -1) {
#if SSSP_MAKING_DEBUG
    printf("%lld has path to %lld\n", city_id, result.nearest_shelter_id);
#endif
    // Copy the predecessor array back to the host
    CUDA_CHECK(cudaMemcpy(h_predecessors, d_predecessors,
                          h_graph.vertex_count * sizeof(ll),
                          cudaMemcpyDeviceToHost));

    // Count path length
    ll path_len = 0;
    ll current = result.nearest_shelter_id;
    while (current != -1) {
      path_len++;
      current = h_predecessors[current];
    }

    // Allocate path array
    result.path = (ll *)malloc(path_len * sizeof(ll));
    result.path_length = path_len;

    // Reconstruct path (in reverse order)
    current = result.nearest_shelter_id;
    ll idx = path_len - 1;
    while (current != -1) {
      result.path[idx--] = current;
      current = h_predecessors[current];
    }
  }
#if SSSP_MAKING_DEBUG
  else {
    printf("%lld has no path\n", city_id);
  }
#endif

  // Free device memory
  CUDA_CHECK(cudaFree(d_distances));
  CUDA_CHECK(cudaFree(d_predecessors));
  CUDA_CHECK(cudaFree(d_updated));
  CUDA_CHECK(cudaFree(d_nearest_shelter));
  CUDA_CHECK(cudaFree(d_min_distance));

  // Free host memory
  free(h_distances);
  free(h_predecessors);

  return result;
}

PathResult *findAllPathsToNearestShelters() {
  // Allocate result array for all cities
  PathResult *results =
      (PathResult *)malloc(h_graph.num_pop_cits * sizeof(PathResult));

  // Process each city
  for (ll i = 0; i < h_graph.num_pop_cits; i++) {
#if SSSP_MAKING_DEBUG
    printf("doing %lld\n", h_graph.walk_stat.id[i]);
#endif

    ll city_id = h_graph.walk_stat.id[i];
    results[i] = findNearestShelterPath(city_id);
  }
  return results;
}
////////////////////////////////////////////////////////////////////

/////////////////////////GRAPH FUNCTIONS////////////////////////////
void make_hgraph(ll v_c, ll e_c, int *edges, ll shelter_count, ll *shelters,
                 ll *shelter_caps, int pop_count, ll *pop_cits,
                 ll *pop_counts) {
  h_graph.vertex_count = v_c;
  h_graph.edge_count = 2 * e_c;
  h_graph.is_shelter = new bool[v_c];
  h_graph.shelter_cap = new ll[v_c];
  h_graph.offsets = new ll[v_c + 1];
  h_graph.dsts = new ll[2 * e_c];
  h_graph.srcs = new ll[2 * e_c];
  h_graph.length = new ll[2 * e_c];
  h_graph.capacity = new ll[2 * e_c];

  // filling city info
  h_graph.num_pop_cits = pop_count;
  h_graph.walk_stat.dist_walked = new ll[pop_count];
  h_graph.walk_stat.id = new ll[pop_count];
  h_graph.walk_stat.old = new ll[pop_count];
  h_graph.walk_stat.young = new ll[pop_count];
  h_graph.walk_stat.curr_city = new ll[pop_count];
  h_graph.walk_stat.curr_road = new ll[pop_count];
  for (ll i = 0; i < pop_count; i++) {
    h_graph.walk_stat.id[i] = pop_cits[i];
    h_graph.walk_stat.curr_city[i] = pop_cits[i];
    h_graph.walk_stat.curr_road[i] = -1;
    h_graph.walk_stat.young[i] = pop_counts[2 * i];
    h_graph.walk_stat.old[i] = pop_counts[2 * i + 1];
    h_graph.walk_stat.dist_walked[i] = 0;
  }

  for (ll i = 0; i < v_c; i++) {
    h_graph.is_shelter[i] = false;
    h_graph.shelter_cap[i] = 0;
  }

  // will use it later forcopying offsets also
  ll *temp_count = new ll[v_c];
  for (ll i = 0; i < v_c; i++) {
    temp_count[i] = 0;
  }

  // edges are of form u1 v1 length capacity
  for (ll i = 0; i < e_c; i++) {
    temp_count[edges[4 * i]]++;
    temp_count[edges[4 * i + 1]]++;
  }

  h_graph.offsets[0] = 0;
  for (ll i = 1; i < v_c + 1; i++) {
    h_graph.offsets[i] = h_graph.offsets[i - 1] + temp_count[i - 1];
  }

  ll temp = 0;
  for (ll i = 0; i < v_c; i++) {
    for (ll j = h_graph.offsets[i]; j < h_graph.offsets[i + 1]; j++) {
      h_graph.srcs[temp++] = i;
    }
    temp_count[i] = h_graph.offsets[i];
  }

  for (ll i = 0; i < e_c; i++) {
#if G_MAKING
    cout << "Making edges " << edges[4 * i] << " ---- " << edges[4 * i + 1];
    cout << "Place to go" << temp_count[edges[4 * i]] << " "
         << temp_count[edges[4 * i + 1]];
    cout << endl;
#endif

    h_graph.dsts[temp_count[edges[4 * i]]] = edges[4 * i + 1];
    h_graph.dsts[temp_count[edges[4 * i + 1]]] = edges[4 * i];

#if G_MAKING
    cout << "Added the following" << endl;
    cout << edges[4 * i] << "----" << h_graph.dsts[temp_count[edges[4 * i]]]
         << endl;
    cout << edges[4 * i + 1] << "----"
         << h_graph.dsts[temp_count[edges[4 * i + 1]]] << endl;
#endif
    h_graph.length[temp_count[edges[4 * i]]] = edges[4 * i + 2];
    h_graph.length[temp_count[edges[4 * i + 1]]] = edges[4 * i + 2];
    h_graph.capacity[temp_count[edges[4 * i + 1]]] = edges[4 * i + 3];
    h_graph.capacity[temp_count[edges[4 * i]]] = edges[4 * i + 3];

    temp_count[edges[4 * i]]++;
    temp_count[edges[4 * i + 1]]++;
  }

  delete[] temp_count;

  for (ll i = 0; i < shelter_count; i++) {
    h_graph.is_shelter[shelters[i]] = true;
    h_graph.shelter_cap[shelters[i]] = shelter_caps[i];
  }
}

void make_dgraph() {
  ll v_c = h_graph.vertex_count;
  ll e_c = h_graph.edge_count;
  int pop_count = h_graph.num_pop_cits;
  d_graph_copy.vertex_count = h_graph.vertex_count;
  d_graph_copy.edge_count = h_graph.edge_count;
  d_graph_copy.num_pop_cits = h_graph.num_pop_cits;

  CUDA_CHECK(cudaMalloc(&d_graph_copy.is_shelter, sizeof(bool) * v_c));
  CUDA_CHECK(cudaMalloc(&d_graph_copy.shelter_cap, sizeof(ll) * v_c));
  CUDA_CHECK(cudaMalloc(&d_graph_copy.offsets, sizeof(ll) * (v_c + 1)));
  CUDA_CHECK(cudaMalloc(&d_graph_copy.dsts, sizeof(ll) * e_c));
  CUDA_CHECK(cudaMalloc(&d_graph_copy.srcs, sizeof(ll) * e_c));
  CUDA_CHECK(cudaMalloc(&d_graph_copy.length, sizeof(ll) * e_c));
  CUDA_CHECK(cudaMalloc(&d_graph_copy.capacity, sizeof(ll) * e_c));

  /////
  CUDA_CHECK(
      cudaMalloc(&d_graph_copy.walk_stat.dist_walked, sizeof(ll) * pop_count));
  CUDA_CHECK(cudaMalloc(&d_graph_copy.walk_stat.id, sizeof(ll) * pop_count));
  CUDA_CHECK(
      cudaMalloc(&d_graph_copy.walk_stat.curr_city, sizeof(ll) * pop_count));
  CUDA_CHECK(
      cudaMalloc(&d_graph_copy.walk_stat.curr_road, sizeof(ll) * pop_count));
  CUDA_CHECK(cudaMalloc(&d_graph_copy.walk_stat.old, sizeof(ll) * pop_count));
  CUDA_CHECK(cudaMalloc(&d_graph_copy.walk_stat.young, sizeof(ll) * pop_count));

  ///////
  CUDA_CHECK(cudaMemcpy(d_graph_copy.is_shelter, h_graph.is_shelter,
                        sizeof(bool) * v_c, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_graph_copy.shelter_cap, h_graph.shelter_cap,
                        sizeof(ll) * v_c, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_graph_copy.offsets, h_graph.offsets,
                        sizeof(ll) * (v_c + 1), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_graph_copy.dsts, h_graph.dsts, sizeof(ll) * e_c,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_graph_copy.srcs, h_graph.srcs, sizeof(ll) * e_c,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_graph_copy.length, h_graph.length, sizeof(ll) * e_c,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_graph_copy.capacity, h_graph.capacity,
                        sizeof(ll) * e_c, cudaMemcpyHostToDevice));
  //////
  CUDA_CHECK(cudaMemcpy(d_graph_copy.walk_stat.id, h_graph.walk_stat.id,
                        sizeof(ll) * pop_count, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_graph_copy.walk_stat.curr_city,
                        h_graph.walk_stat.curr_city, sizeof(ll) * pop_count,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_graph_copy.walk_stat.curr_road,
                        h_graph.walk_stat.curr_road, sizeof(ll) * pop_count,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_graph_copy.walk_stat.young, h_graph.walk_stat.young,
                        sizeof(ll) * pop_count, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_graph_copy.walk_stat.old, h_graph.walk_stat.old,
                        sizeof(ll) * pop_count, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_graph_copy.walk_stat.dist_walked,
                        h_graph.walk_stat.dist_walked, sizeof(ll) * pop_count,
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpyToSymbol(d_graph, &d_graph_copy, sizeof(Graph), 0,
                                cudaMemcpyHostToDevice));
}

void delete_hgraph() {
  delete[] h_graph.is_shelter;
  delete[] h_graph.shelter_cap;
  delete[] h_graph.dsts;
  delete[] h_graph.srcs;
  delete[] h_graph.length;
  delete[] h_graph.capacity;
  delete[] h_graph.offsets;

  ////
  delete[] h_graph.walk_stat.dist_walked;
  delete[] h_graph.walk_stat.id;
  delete[] h_graph.walk_stat.curr_city;
  delete[] h_graph.walk_stat.curr_road;
  delete[] h_graph.walk_stat.young;
  delete[] h_graph.walk_stat.old;
}

void delete_dgraph() {
  CUDA_CHECK(cudaFree(d_graph_copy.is_shelter));
  CUDA_CHECK(cudaFree(d_graph_copy.shelter_cap));
  CUDA_CHECK(cudaFree(d_graph_copy.offsets));
  CUDA_CHECK(cudaFree(d_graph_copy.dsts));
  CUDA_CHECK(cudaFree(d_graph_copy.srcs));
  CUDA_CHECK(cudaFree(d_graph_copy.length));
  CUDA_CHECK(cudaFree(d_graph_copy.capacity));

  ////
  CUDA_CHECK(cudaFree(d_graph_copy.walk_stat.id));
  CUDA_CHECK(cudaFree(d_graph_copy.walk_stat.curr_city));
  CUDA_CHECK(cudaFree(d_graph_copy.walk_stat.curr_road));
  CUDA_CHECK(cudaFree(d_graph_copy.walk_stat.young));
  CUDA_CHECK(cudaFree(d_graph_copy.walk_stat.old));
  CUDA_CHECK(cudaFree(d_graph_copy.walk_stat.dist_walked));
}
////////////////////////////////////////////////////////////////////

///////////////////testing kernels//////////////////////////////////
__global__ void check_d_graph() {
  for (ll i = 0; i < d_graph.edge_count; i++) {
    printf("%ld ", d_graph.dsts[i]);
  }
  printf("\n");
  for (ll i = 0; i < d_graph.edge_count; i++) {
    printf("%ld ", d_graph.srcs[i]);
  }
  printf("\n");

  for (ll i = 0; i < d_graph.vertex_count + 1; i++) {
    printf("%ld ", d_graph.offsets[i]);
  }
  printf("\n");
}

////////////////////////////////////////////////////////////////////

///////////////////////MAKE OUTPUTS DIRECTLY KERNELS////////////////
PathResult *copy_path_device(PathResult *h_route) {
  PathResult *d_paths;
  // Allocate memory for the array of PathResult structures
  CUDA_CHECK(cudaMalloc(&d_paths, sizeof(PathResult) * h_graph.num_pop_cits));

  // Create a temporary array to prepare the data
  PathResult *temp_paths =
      (PathResult *)malloc(sizeof(PathResult) * h_graph.num_pop_cits);

  for (ll i = 0; i < h_graph.num_pop_cits; i++) {
    // Copy basic fields
    temp_paths[i].distance = h_route[i].distance;
    temp_paths[i].path_length = h_route[i].path_length;
    temp_paths[i].nearest_shelter_id = h_route[i].nearest_shelter_id;

    // Allocate device memory for path array
    ll *d_path = NULL;
    if (h_route[i].path != NULL && h_route[i].path_length > 0) {
      CUDA_CHECK(cudaMalloc(&d_path, sizeof(ll) * h_route[i].path_length));
      // Copy path data
      CUDA_CHECK(cudaMemcpy(d_path, h_route[i].path,
                            sizeof(ll) * h_route[i].path_length,
                            cudaMemcpyHostToDevice));
    }
    // Store device pointer in temp structure
    temp_paths[i].path = d_path;
  }

  // Copy the entire array of prepared structures to device
  CUDA_CHECK(cudaMemcpy(d_paths, temp_paths,
                        sizeof(PathResult) * h_graph.num_pop_cits,
                        cudaMemcpyHostToDevice));

  // Clean up
  free(temp_paths);

  return d_paths;
}

// to find which city is dropping how much peeps at its shelter
__global__ void drop_at_shelter_info(PathResult *d_route,
                                     ll *drop_in_shelters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < d_graph.num_pop_cits) {
    ll pops;
    if (d_route[idx].distance > elder_max_dist) {
      pops = d_graph.walk_stat.young[idx];
    } else {
      pops = d_graph.walk_stat.young[idx] + d_graph.walk_stat.old[idx];
    }

    ll my_shelt = d_route[idx].nearest_shelter_id;
    ll to_drop = 0;

    while (d_graph.shelter_cap[my_shelt] > 0) {
      ll available = d_graph.shelter_cap[my_shelt];

      to_drop = min(available, pops);
      ll after_drop = available - to_drop;

      ll found = atomicCAS((unsigned long long *)&d_graph.shelter_cap[my_shelt],
                           (unsigned long long)available,
                           (unsigned long long)after_drop);

      if (found == available) {
        break;
      }
    }
    drop_in_shelters[idx] = to_drop;
  }
}

void drop_people_off(PathResult *h_route, ll *&path_size, ll **&paths,
                     ll *&num_drops, ll ***&drops) {
  PathResult *d_route = copy_path_device(h_route);
  int block_size = BLOCK_SIZE;
  int grih_size = (h_graph.num_pop_cits + block_size - 1) / block_size;
  ll *shelter_drop = new ll[h_graph.num_pop_cits];
  ll *d_shelter_drop;
  CUDA_CHECK(cudaMalloc(&d_shelter_drop, sizeof(ll) * h_graph.num_pop_cits));
  CUDA_CHECK(cudaMemset(d_shelter_drop, 0, sizeof(ll) * h_graph.num_pop_cits));

  drop_at_shelter_info<<<grih_size, block_size>>>(d_route, d_shelter_drop);
  CUDA_CHECK(cudaGetLastError());

  path_size = (ll *)malloc(sizeof(ll) * h_graph.num_pop_cits);
  paths = (ll **)malloc(sizeof(ll *) * h_graph.num_pop_cits);

  for (ll i = 0; i < h_graph.num_pop_cits; i++) {
    path_size[i] = h_route[i].path_length;
  }

  for (ll i = 0; i < h_graph.num_pop_cits; i++) {
    paths[i] = (ll *)malloc(sizeof(ll) * h_route[i].path_length);
    for (ll j = 0; j < path_size[i]; j++) {
      paths[i][j] = h_route[i].path[j];
    }
  }

  num_drops = (ll *)malloc(sizeof(ll) * h_graph.num_pop_cits);
  for (ll i = 0; i < h_graph.num_pop_cits; i++) {
    num_drops[i] = 2;
  }

  drops = (ll ***)malloc(sizeof(ll **) * h_graph.num_pop_cits);

  CUDA_CHECK(cudaMemcpy(shelter_drop, d_shelter_drop,
                        sizeof(ll) * h_graph.num_pop_cits,
                        cudaMemcpyDeviceToHost));

  for (ll i = 0; i < h_graph.num_pop_cits; i++) {
    ll final_drop = shelter_drop[i];
    ll total = h_graph.walk_stat.young[i] + h_graph.walk_stat.old[i];
    ll initial_young_drp = 0;
    ll initial_old_drp = 0;
    ll final_young_drp = h_graph.walk_stat.young[i];
    ll final_old_drp = h_graph.walk_stat.old[i];

    // prioratise younglings, then old peeps
    // check if we can't carry all yongs itself
    if (final_drop < final_young_drp) {
      initial_old_drp = final_old_drp;
      total -= final_old_drp;
      final_old_drp = 0;

      initial_young_drp = final_young_drp - final_drop;
      final_young_drp = final_drop;

    } else if (total > final_drop >= final_young_drp) {

      // difference will be old people that cant be accomodated
      initial_old_drp = total - final_drop;
      final_old_drp -= initial_old_drp;
    } else {
      // nothing actually, everybody can be dropped then
    }

    drops[i] = (ll **)malloc(sizeof(ll *) * num_drops[i]);

    for (ll j = 0; j < num_drops[i]; j++) {
      drops[i][j] = (ll *)malloc(sizeof(ll) * 3);
      if (j == 0) {
        drops[i][j][0] = h_route[i].path[0];
        drops[i][j][1] = initial_young_drp;
        drops[i][j][2] = initial_old_drp;
      } else {
        drops[i][j][0] = h_route[i].nearest_shelter_id;
        drops[i][j][1] = final_young_drp;
        drops[i][j][2] = final_old_drp;
      }
    }
  }

  for (ll i = 0; i < h_graph.num_pop_cits; i++) {
    cudaFree(h_route[i].path);
  }
  cudaFree(h_route);
}
////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
    return 1;
  }

  ifstream infile(argv[1]); // Read input file from command-line argument
  if (!infile) {
    std::cerr << "Error: Cannot open file " << argv[1] << "\n";
    return 1;
  }

  long long num_cities;
  infile >> num_cities;

  long long num_roads;
  infile >> num_roads;

  // Store roads as a flat array: [u1, v1, length1, capacity1, u2, v2, length2,
  // capacity2, ...]
  int *roads = new int[num_roads * 4];

  for (int i = 0; i < num_roads; i++) {
    infile >> roads[4 * i] >> roads[4 * i + 1] >> roads[4 * i + 2] >>
        roads[4 * i + 3];
  }

  int num_shelters;
  infile >> num_shelters;

  // Store shelters separately
  long long *shelter_city = new long long[num_shelters];
  long long *shelter_capacity = new long long[num_shelters];

  for (int i = 0; i < num_shelters; i++) {
    infile >> shelter_city[i] >> shelter_capacity[i];
  }

  int num_populated_cities;
  infile >> num_populated_cities;

  // Store populated cities separately
  long long *city = new long long[num_populated_cities];
  long long *pop = new long long[num_populated_cities *
                                 2]; // Flattened [prime-age, elderly] pairs

  for (long long i = 0; i < num_populated_cities; i++) {
    infile >> city[i] >> pop[2 * i] >> pop[2 * i + 1];
  }

  int max_distance_elderly;
  infile >> max_distance_elderly;

  infile.close();

  // set your answer to these variables
  long long *path_size;
  long long **paths;
  long long *num_drops;
  long long ***drops;

  ///////////My Computation/////////////
  make_hgraph(num_cities, num_roads, roads, num_shelters, shelter_city,
              shelter_capacity, num_populated_cities, city, pop);
#if CSR_DEBUG
  printf("\n\n");
  printf("Host Side Graph\n\n");

  cout << "Graph CSR" << endl;

  cout << h_graph.vertex_count << " vertices and " << h_graph.edge_count
       << " edges" << endl;

  cout << "DSTS: ";
  for (ll i = 0; i < h_graph.edge_count; i++) {
    cout << h_graph.dsts[i] << " ";
  }
  cout << endl;

  cout << "SRCS: ";
  for (ll i = 0; i < h_graph.edge_count; i++) {
    cout << h_graph.srcs[i] << " ";
  }
  cout << endl;

  cout << "EDGE: ";
  for (ll i = 0; i < h_graph.edge_count; i++) {
    cout << i << " ";
  }
  cout << endl;

  for (ll i = 0; i < h_graph.edge_count; i++) {
    cout << h_graph.length[i] << " ";
  }
  cout << endl;

  for (ll i = 0; i < h_graph.edge_count; i++) {
    cout << h_graph.capacity[i] << " ";
  }
  cout << endl;

  for (ll i = 0; i < h_graph.vertex_count + 1; i++) {
    cout << h_graph.offsets[i] << " ";
  }
  cout << endl;

  cout << "Shelter Info" << endl;
  for (ll i = 0; i < h_graph.vertex_count; i++) {
    printf("%d ", h_graph.is_shelter[i]);
  }
  printf("\n");
  for (ll i = 0; i < h_graph.vertex_count; i++) {
    printf("%lld ", h_graph.shelter_cap[i]);
  }
  printf("\n");

  cout << "Pop info" << endl;
  for (ll i = 0; i < h_graph.num_pop_cits; i++) {
    printf("%lld ", h_graph.walk_stat.id[i]);
  }
  printf("\n");

  for (ll i = 0; i < h_graph.num_pop_cits; i++) {
    printf("%lld ", h_graph.walk_stat.young[i]);
  }
  printf("\n");
  for (ll i = 0; i < h_graph.num_pop_cits; i++) {
    printf("%lld ", h_graph.walk_stat.old[i]);
  }
  printf("\n");
  for (ll i = 0; i < h_graph.num_pop_cits; i++) {
    printf("%lld ", h_graph.walk_stat.dist_walked[i]);
  }
  printf("\n");
#endif

  make_dgraph();
  CUDA_CHECK(cudaMemcpyToSymbol(elder_max_dist, &max_distance_elderly,
                                sizeof(int), 0, cudaMemcpyHostToDevice));

#if CSR_D_DEBUG
  printf("\n\n");
  printf("Device Side Graph\n\n");
  check_d_graph<<<1, 1>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
#endif

#if SSSP_MAKING_DEBUG
  printf("\n\n");
  printf("Doing SSSP\n\n");
#endif

  // Find paths to nearest shelters
  PathResult *results = findAllPathsToNearestShelters();

#if SSSP_DEBUG
  // Print results
  for (int i = 0; i < h_graph.num_pop_cits; i++) {
    printf("City %lld: Nearest shelter is %lld at distance %lld\n",
           h_graph.walk_stat.id[i], results[i].nearest_shelter_id,
           results[i].distance);

    printf("Path: ");
    for (int j = 0; j < results[i].path_length; j++) {
      printf("%lld ", results[i].path[j]);
    }
    printf("\n\n");
  }
#endif

  // main_sim(results);
  // CUDA_CHECK(cudaDeviceSynchronize());
  drop_people_off(results, path_size, paths, num_drops, drops);

  for (ll i = 0; i < h_graph.num_pop_cits; i++) {
    // Free path memory
    free(results[i].path);
  }
  free(results);

  //////////////////////////////////////

  ofstream outfile(argv[2]); // Read input file from command-line argument
  if (!outfile) {
    std::cerr << "Error: Cannot open file " << argv[2] << "\n";
    return 1;
  }
  for (long long i = 0; i < num_populated_cities; i++) {
    long long currentPathSize = path_size[i];
    for (long long j = 0; j < currentPathSize; j++) {
      outfile << paths[i][j] << " ";
    }
    outfile << "\n";
  }

  for (long long i = 0; i < num_populated_cities; i++) {
    long long currentDropSize = num_drops[i];
    for (long long j = 0; j < currentDropSize; j++) {
      for (int k = 0; k < 3; k++) {
        outfile << drops[i][j][k] << " ";
      }
    }
    outfile << "\n";
  }

  return 0;
}