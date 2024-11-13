#include <cuda_runtime.h>
#include <iostream>

__global__ void confusion_matrix_kernel(int* d_predictions, int* d_labels, int* d_confusion_matrix, int num_classes, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_samples) {
        int pred = d_predictions[idx];
        int label = d_labels[idx];

        if (pred < num_classes && label < num_classes) {
            atomicAdd(&d_confusion_matrix[label * num_classes + pred], 1);
        }
    }
}

void calculate_confusion_matrix(int* d_predictions, int* d_labels, int* d_confusion_matrix, int num_classes, int num_samples) {
    // Initialize confusion matrix to zero on the GPU
    cudaMemset(d_confusion_matrix, 0, num_classes * num_classes * sizeof(int));

    // Launch kernel to compute confusion matrix
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_samples + threadsPerBlock - 1) / threadsPerBlock;

    confusion_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, d_labels, d_confusion_matrix, num_classes, num_samples);

    // Synchronize to ensure all threads are done
    cudaDeviceSynchronize();
}

int main() {
    const int num_samples = 28;
    const int num_classes = 2;

    // Allocate and initialize host memory
    int h_predictions[num_samples] = {1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0}; // Fill this with your data
    int h_labels[num_samples] = {1,0,1,0,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1,1,1};      // Fill this with your data

    // Allocate device memory
    int* d_predictions;
    int* d_labels;
    int* d_confusion_matrix;

    cudaMalloc(&d_predictions, num_samples * sizeof(int));
    cudaMalloc(&d_labels, num_samples * sizeof(int));
    cudaMalloc(&d_confusion_matrix, num_classes * num_classes * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_predictions, h_predictions, num_samples * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, num_samples * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate confusion matrix
    calculate_confusion_matrix(d_predictions, d_labels, d_confusion_matrix, num_classes, num_samples);

    // Allocate host memory for the confusion matrix and copy result back
    int h_confusion_matrix[num_classes * num_classes];
    cudaMemcpy(h_confusion_matrix, d_confusion_matrix, num_classes * num_classes * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the confusion matrix
    std::cout << "Confusion Matrix:" << std::endl;
    for (int i = 0; i < num_classes; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            std::cout << h_confusion_matrix[i * num_classes + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_predictions);
    cudaFree(d_labels);
    cudaFree(d_confusion_matrix);

    return 0;
}
