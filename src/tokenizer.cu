#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include <cstdio> 
#include <algorithm>  
__global__ void tokenize(int* list_of_bytes, int* hash_map, int interval_length,
                         int vocab_size, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int start = idx * interval_length;
  int end = start + interval_length;
  if (start >= size) return; 
  if (end > size) end = size;
  int i = start;
  while (i < end) {
    while (i < end && list_of_bytes[i] == -1) i++;
    if (i >= end) break;
    int j = i + 1;
    while (j < end && list_of_bytes[j] == -1) j++;
    if (j >= end) break;
    int a = list_of_bytes[i];
    int b = list_of_bytes[j];
    if (a >= 0 && a < vocab_size && b >= 0 && b < vocab_size) {
      atomicAdd(&hash_map[a * vocab_size + b], 1);
    }
    i = j;
  }
}
__global__ void maxReduceBlock(int* hash_map, int N, int* result, int vocab_size) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  int maxVal = -1;
  int maxIdx = -1;
  while (i < N) {
    int v = hash_map[i];
    if (v > maxVal) { maxVal = v; maxIdx = i; }
    i += blockDim.x * gridDim.x;
  }
  sdata[tid] = maxVal;
  sdata[tid + blockDim.x] = maxIdx;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sdata[tid] < sdata[tid + s]) {
        sdata[tid] = sdata[tid + s];
        sdata[tid + blockDim.x] = sdata[tid + s + blockDim.x];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    int winner = sdata[blockDim.x];
    result[blockIdx.x * 3] = sdata[0];
    result[blockIdx.x * 3 + 1] = winner / vocab_size;
    result[blockIdx.x * 3 + 2] = winner % vocab_size;
  }
}

__global__ void change_pair(int* list_of_bytes,
                            int interval_length,
                            int first_pair, int second_pair, int id, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int start = idx * interval_length;
  int end = start + interval_length;
  if (start >= size) return;
  if (end > size) end = size;
  int i = start;
  while (i < end) {
    while (i < end && list_of_bytes[i] == -1) i++;
    if (i >= end) break;
    int j = i + 1;
    while (j < end && list_of_bytes[j] == -1) j++;
    if (j >= end) break;
    if (list_of_bytes[i] == first_pair && list_of_bytes[j] == second_pair) {
      list_of_bytes[i] = id;
      list_of_bytes[j] = -1;
      i++;
    } else {
      i++; 
    }
  }
}

int* readFileToBytesList(const char* filename, int* size) {
  FILE* file = fopen(filename, "rb");
  if (!file) {
    printf("Error: Could not open file %s\n", filename);
    return nullptr;
  }

  fseek(file, 0, SEEK_END);
  *size = ftell(file);
  fseek(file, 0, SEEK_SET);

  unsigned char* file_bytes = (unsigned char*)malloc(*size);
  if (!file_bytes) {
    printf("Error: Could not allocate memory\n");
    fclose(file);
    return nullptr;
  }

  size_t bytes_read = fread(file_bytes, 1, *size, file);
  fclose(file);

  if (bytes_read != *size) {
    printf("Error: Could not read complete file\n");
    free(file_bytes);
    return nullptr;
  }
  int* bytes_list = (int*)malloc(*size * sizeof(int));
  if (!bytes_list) {
    printf("Error: host allocation failed for %d ints\n", *size);
    free(file_bytes);
    return nullptr;
  }
  for (int i = 0; i < *size; i++) bytes_list[i] = (int)file_bytes[i];
  free(file_bytes);
  return bytes_list;
}
bool saveMergesHistory(const char* filename, int* merges_history,
                       int num_merges) {
  FILE* file = fopen(filename, "w");
  if (!file) {
    printf("Error: Could not open file %s for writing\n", filename);
    return false;
  }

  for (int i = 0; i < num_merges; i++) {
    fprintf(file, "%d %d\n", merges_history[i * 2], merges_history[i * 2 + 1]);
  }

  fclose(file);
  printf("Successfully saved %d merge operations to %s\n", num_merges,
         filename);
  return true;
}
#define CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(1); \
  } \
} while (0)

int main() {
  const char* filename = "file_bigger.txt";
  int threads_per_block = 128;
  int max_blocks = 256; 
  int* size = new int;        
  int* h_list = readFileToBytesList(filename, size);
  if (!h_list) return 1;

  int vocab_size = 20000 - 256;
  size_t table_elems = (size_t)vocab_size * (size_t)vocab_size;
  size_t bytes_needed = table_elems * sizeof(int);
  int interval_length = 1 + (*size / (threads_per_block * max_blocks));
  int total_intervals = (*size + interval_length - 1) / interval_length;
  int blocks = std::min(max_blocks, (total_intervals + threads_per_block - 1) / threads_per_block);

  int max_merges = vocab_size;
  int* merges_history = new int[max_merges * 2];

  int* d_list = nullptr;
  int* d_hash_table = nullptr;
  int* d_local_max = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_list, *size * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_list, h_list, *size * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc((void**)&d_hash_table, bytes_needed));
  CUDA_CHECK(cudaMemset(d_hash_table, 0, bytes_needed));

  int iterations = vocab_size;
  int best_count = -1;
  int first_pair = 0;
  int second_pair = 0;
  int id = 256;
  CUDA_CHECK(cudaMalloc((void**)&d_local_max, 3 * sizeof(int) * blocks));
  int* h_local_max = (int*)malloc(3 * sizeof(int) * blocks);
  if (!h_local_max) {
    fprintf(stderr, "Host allocation failed for local max buffer\n");
    return 1;
  }

  while (iterations > 0) {
    CUDA_CHECK(cudaMemset(d_hash_table, 0, bytes_needed));

    tokenize<<<blocks, threads_per_block>>>(d_list, d_hash_table, interval_length, vocab_size, *size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t shared_bytes = 2 * threads_per_block * sizeof(int);
    maxReduceBlock<<<blocks, threads_per_block, shared_bytes>>>(d_hash_table, (int)table_elems, d_local_max, vocab_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    best_count = -1;
    CUDA_CHECK(cudaMemcpy(h_local_max, d_local_max, 3 * sizeof(int) * blocks, cudaMemcpyDeviceToHost));
    for (int bi = 0; bi < blocks; bi++) {
      int c = h_local_max[bi * 3];
      if (c > best_count) {
        best_count = c;
        first_pair = h_local_max[bi * 3 + 1];
        second_pair = h_local_max[bi * 3 + 2];
      }
    }
    if (best_count <= 0) {
      printf("No more frequent pairs found. Stopping early at iteration %d.\n", id);
      break;
    }
    if (id % 50 == 0) {
      printf("Top pair at merge %d: (%d, %d) count=%d\n", id, first_pair, second_pair, best_count);
    }
    if (id - 256 < max_merges) {
      merges_history[2 * (id - 256)] = first_pair;
      merges_history[2 * (id - 256) + 1] = second_pair;
    }
  

  change_pair<<<blocks, threads_per_block>>>(d_list, interval_length, first_pair, second_pair, id, *size);
  id++;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    iterations--;
    if (iterations % 10 == 0) {
      printf("Iterations left (cap): %d\n", iterations);
    }
  }

  saveMergesHistory("merges_history.txt", merges_history, id - 256);

  CUDA_CHECK(cudaFree(d_list));
  CUDA_CHECK(cudaFree(d_local_max));
    CUDA_CHECK(cudaFree(d_hash_table));
    free(h_local_max);
    free(h_list);
    delete[] merges_history;
    delete size;
    return 0;
  }


