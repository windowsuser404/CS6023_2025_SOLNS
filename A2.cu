#include <chrono>
#include <cuda.h>
// #include <cuda/cuda_runtime.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdio.h>

using namespace std;

using std::cin;
using std::cout;

#define DEBUG 1
using Lint = long int;

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

__global__ void dkernel(long int *matrix, long int *filter, long int *result,
                        int h, int w, int c, int r, int s, int k) {
  // sample kernel you can use your own kernel
  // making filter shared memory
  __shared__ Lint D_filter[4096];
  Lint total_pix_per_img = h * w;
  Lint local_id = threadIdx.z * blockDim.x * blockDim.y +
                  threadIdx.x * blockDim.y + threadIdx.y;
  Lint block_id = blockIdx.x;
  Lint thread_per_blk = blockDim.z * blockDim.y * blockDim.x;
  Lint global_id = block_id * thread_per_blk + local_id;

  Lint full_filter_size = r * s * c;
  Lint per_channel_filter_size = r * s;

  // the quotient gives which image u r in, which is the filter we care about
  // Lint filter_id = global_id / (total_pix_per_img);
  Lint filter_id = blockIdx.y;
  Lint filter_base = filter_id * full_filter_size;

  // load each channel
  // increase by total_threads for coaslesce, limit to one filter size
  for (Lint element = local_id; element < r * s * c;
       element += thread_per_blk) {
    Lint element_index = element;
    // each block is to calculate one filter
    D_filter[element_index] = filter[filter_base + element_index];
  }

  // ensures all filter elements are loaded
  __syncthreads();

  // time to do computations

  // img_idx is line, what is my index in all thread for a particular filter_id
  // Lint img_indx = global_id % total_pix_per_img;
  // assuming total threads in x alone is enough
  Lint img_indx = global_id;
  // now, we have H and W, we should have launched H*W threads, im taking row
  // major, so divide by column to get the respictive row and col no
  Lint my_row = img_indx / w;
  Lint my_col = img_indx % w;

  if (my_row < h && my_col < w) {
    // ensure r and s are odd numbers lol
    Lint center_row = r / 2;
    Lint center_col = s / 2;

    Lint sum = 0;

    for (Lint channel = 0; channel < c; channel++) {
      Lint img_base = channel * total_pix_per_img;
      Lint filter_base = channel * per_channel_filter_size;
      for (Lint row = 0; row < r; row++) {
        Lint input_row = my_row - center_row + row;
        for (Lint col = 0; col < s; col++) {
          Lint input_col = my_col - center_col + col;
          if ((0 <= input_row && input_row < h) &&
              (0 <= input_col && input_col < w)) {
            Lint img_index = img_base + input_row * w + input_col;
            Lint filter_index = filter_base + row * s + col;
            sum += matrix[img_index] * D_filter[filter_index];
          }
        }
      }
    }
    Lint out_base = filter_id * total_pix_per_img;
    Lint out_index = out_base + my_row * w + my_col;

    result[out_index] = sum;
  }
}

int main(int argc, char **argv) {
  int h, w, c;
  cin >> h >> w >> c;
  long int *h_mat = new long int[h * w * c];
  for (long int i = 0; i < h * w * c; i++) {
    cin >> h_mat[i];
  }

  int cf, r, s, k;
  cin >> cf >> r >> s >> k;

  long int *h_filter = new long int[r * s * c * k];
  for (long int i = 0; i < r * s * c * k; i++) {
    cin >> h_filter[i];
  }
  long int *h_ans = new long int[h * w * k];

  /**
   *
   * DO NOT CHANGE ANYTHING ABOVE THIS LINE
   *
   **/

  auto start = std::chrono::high_resolution_clock::now(); // keep it just before
                                                          // the kernel launch

  /****************************************************Start
   * Here***********************************************************/

  /**
      Do device allocations, kernel launches and copying everything here
      and the final answer should be stored back in h_ans, use cudaFree to free
     up the allocated memory on GPU
  */

  // vairables
  Lint *d_mat;
  Lint *d_filter;
  Lint *d_result;

  // mallocs
  Lint input_size = sizeof(Lint) * h * w * c;
  Lint filter_size = sizeof(Lint) * k * r * s * c;
  Lint output_size = sizeof(Lint) * h * w * k;
  cudaMalloc(&d_mat, input_size);
  cudaMalloc(&d_filter, filter_size);
  cudaMalloc(&d_result, output_size);

  // copies
  cudaMemcpy(d_mat, h_mat, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);

  // call kernel
  dim3 blockshape = dim3(1024, 1, 1);
  dim3 gridshape = dim3((h * w + 1023) / 1024, k, 1);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
  dkernel<<<gridshape, blockshape>>>(d_mat, d_filter, d_result, h, w, c, r, s,
                                     k);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
  // copy result
  cudaMemcpy(h_ans, d_result, output_size, cudaMemcpyDeviceToHost);

  /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is
   * stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
  auto end = std::chrono::high_resolution_clock::now(); // keep it just after
                                                        // the kernel launch
  std::chrono::duration<double> elapsed1 = end - start;
  /**
   *
   * DO NOT CHANGE ANYTHING BELOW THIS LINE
   *
   */

  cudaDeviceSynchronize();
  std::ofstream file("cuda.out");
  if (file.is_open()) {
    for (long int i = 0; i < h * k; i++) {
      for (long int j = 0; j < w; j++) {
        file << h_ans[i * w + j] << " ";
      }
      file << "\n";
    }
    file.close();
  } else {
    std::cout << "Unable to open file";
  }

  std::ofstream file2("cuda_timing.out");
  if (file2.is_open()) {
    file2 << elapsed1.count() << "\n";
    file2.close();
  } else {
    std::cout << "Unable to open file";
  }

  return 0;
}
