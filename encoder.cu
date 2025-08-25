#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include <cstdio> 
#include <fstream>
#include <iostream>
#include <algorithm>  
#include <numeric>
using namespace std;


__global__ void encoder(int size, int workers, int* process, int* merge_data, int n, int iter)
{
    int interval = (size + workers - 1) / workers;
    int start = blockIdx.x * blockDim.x * interval + threadIdx.x * interval;
    int end = blockIdx.x * blockDim.x * interval + (threadIdx.x +1 ) * interval;
    int j, k;
    for(int i = iter * 10; i < iter * 10 + 10; i++){
    if(iter == n){
      break;
    }
    j = start;
    k = start;
    while(j < end && j < size && k < end && k < size) {
        if(process[j] != -1) {
            k = j + 1;
            while(k < size && k < end && process[k] == -1) k++;
            if(process[j] == merge_data[2 * i] && process[k] == merge_data[2 * i + 1] && process[j] != -1 && process[k] != -1) {
            process[j] = -1;
            process[k] = 256 + i;
        }
        j = k;
        } else {
          j++;
        }
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
  int* bytes_list;
  cudaMallocManaged(&bytes_list, *size * sizeof(int));
  if (!bytes_list) {
    printf("Error: host allocation failed for %d ints\n", *size);
    free(file_bytes);
    return nullptr;
  }
  for (int i = 0; i < *size; i++) bytes_list[i] = (int)file_bytes[i];
  free(file_bytes);
  return bytes_list;
}
int* read_merges(const char* filename, int n){
  int* array;
  cudaMallocManaged(&array, n * 2 * sizeof(int));
  std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file\n";
        return nullptr;
    }

    int a, b, i = 0;
    while (file >> a >> b) {
        array[i++] = a;
        array[i++] = b;
    }
    file.close();
    return array;
}
void save_to_file(int* array, int n, const char* filename) {
  std::ofstream file(filename, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
      std::cerr << "Error: could not open file for writing\n";
      return;
  }

  for (int i = 0; i < n; i++) {
      if(array[i] != -1){
      file << array[i] << "\n";
      }
  }

  file.close();
}

int main(void){
    int size;
    int n = 19488;
    const char* filename = "data/shard.txt";
    const char* merge_filename = "merges_history.txt";
    int* process = readFileToBytesList(filename, &size);
    int* merge_data = read_merges(merge_filename, n);
    int threadsPerBlock = 256;
    int blocks = 32;
    printf("Launching encoder kernel with %d blocks and %d threads per block\n", blocks, threadsPerBlock);
    for(int iter = 0; iter< (n + 10 - 1) / 10; iter++){
    encoder<<<blocks, threadsPerBlock>>>(size, blocks * threadsPerBlock, process, merge_data, n, iter);
    cudaDeviceSynchronize();
    printf("%d\n", iter);
    }
    save_to_file(process, size, "data/output.txt");
    printf("end");
    cudaFree(process);
    cudaFree(merge_data);
    return 0;
}