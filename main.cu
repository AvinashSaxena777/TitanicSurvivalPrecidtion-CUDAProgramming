#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <numeric>
#include <stdlib.h>
#include <vector>
#include <algorithm>
using namespace std;
#define REDUCE_BLOCK_SIZE 128

struct Matrix {
	Matrix() : elements(NULL), width(0), height(0), pitch(0) {}
	~Matrix() { if (elements) delete[] elements; }
	unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
};

__global__ void kernelMatrixMul(float*, float*, float*, int, int, int, int);
__global__ void kernelSigmoid(float*, int);
__global__ void kernelMatrixAbsError(float*, float*, float*, int, int);
__global__ void kernelAbsError(float*, float*, float*, int);
__global__ void kernelUpdateParams(float*, float*, float*, float*, int, float);
__global__ void kernelCrossEntropy(float*, float*, float*, int);
__global__ void kernelReduction(float*, float*, int);
__global__ void confusion_matrix_kernel(float*, float*, int*, int, int);

inline static void InitializeMatrix(Matrix *mat, int x, int y, float val) {
	if (x > mat->width || y > mat->height) {
		throw ("invalid access - Initialize Matrix");
	}
	mat->elements[y * mat->width + x] = val;
}

inline static float Matrix_Element_Required(Matrix *mat, int x, int y)
{
	if (x > mat->width || y > mat->height) {
		throw ("invalid access - Matrix Element Required");
	}
	return mat->elements[y * mat->width + x];
}

static void AllocateMatrix(Matrix *mat, int height, int width)
{
	mat->elements = new float[height * width];
	mat->width = width;
	mat->height = height;
	for (int i = 0; i < mat->width; i++) {
		for (int j = 0; j < mat->height; j++) {
			InitializeMatrix(mat, i, j, 0.0f);
		}
	}
}

static void DisplayMatrix(Matrix &mat, bool force = false)
{
	std::cout << "Dim: " << mat.height << ", " << mat.width << "\n";
	if ((mat.width < 10 && mat.height < 10) || force)
	{
		for (int j = 0; j < mat.height; j++) {
			for (int i = 0; i < mat.width; i++) {
				std::cout << Matrix_Element_Required(&mat, i, j) << "\t";
			}
			std::cout << "\n";
		}
	}
	std::cout << std::endl;
}

static bool readData(string file_name, Matrix *X, Matrix *y) {

	ifstream s(file_name.c_str());
	//ifstream s(file_name);
	if (!s.is_open()) {
		//throw runtime_error(file_name + " doesn't exist");
		printf("The file does not exist\n");
	}

	int rows = 0;
	int cols = 0;
	string line;
	while (getline(s, line)) {
		// if we read first line, check how many columns
		if (rows++ == 0) {
			stringstream ss(line);

			while (ss.good()) {
				string substr;
				getline(ss, substr, ',');
				cols++;
			}
		}
	}
	std::cout << "Found " << rows << " rows with " << cols << " columns." << std::endl;
	s.clear() ;
	s.seekg(0, ios::beg);

	AllocateMatrix (X, rows - 1,cols - 2);
	AllocateMatrix (y, rows - 1, 1);

	// go to second line
	getline(s, line);
	int ya = 0;
	while (getline(s, line)) {
		stringstream ss(line);

		int xa = 0;
		while (ss.good()) {
			string substr;
			getline(ss, substr, ',');
			// first column is uninteresting
			// second column is target values
			if (xa == 1) {
				float val = atof(substr.c_str());
				InitializeMatrix(y, 0, ya, val);
			} else if (xa > 1) {
				float val = atof(substr.c_str());
				InitializeMatrix(X, (xa - 2), ya, val);
			}
			xa++;
		}
		ya++;
	}

	return true;
}

static void minMaxNormalization(Matrix *m)
{
	for (int x = 0; x < m->width; ++x) {
		// calculate std for each column
		float min = Matrix_Element_Required(m, x, 0);
		float max = Matrix_Element_Required(m, x, 0);
		for (int y = 1; y < m->height; ++y) {
			float val = Matrix_Element_Required(m, x, y);
			if (val < min) {
				min = val;
			} else if (val > max) {
				max = val;
			}
		}

		for (int y = 0; y < m->height; ++y) {
			float val = Matrix_Element_Required(m, x, y);
			InitializeMatrix(m, x, y, (val - min) / max);
		}
	}
}

static void InitializeRandom(Matrix *mat, float LO, float HI)
{
	for (int i = 0; i < mat->width; ++i) {
		for (int j = 0; j < mat->height; ++j) {
			float r = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
			InitializeMatrix(mat, i, j, r);
		}
	}
}

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
#define SAFE_CALL(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__global__ void kernelMatrixMul(float *m1, float *m2, float *r, int m1w, int m2w, int rw, int rh)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < rh) && (col < rw)) {
		// dot product
		float accum = 0.0f;
		for (int c = 0; c < m1w; c++)
		{
			float v1 = m1[row * m1w + c];
			float v2 = m2[c * m2w + col];
			accum += (v1 *  v2);
		}

		r[row * rw + col] = accum;
	}
}

__global__ void kernelSigmoid(float *r, int m)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < m) {
		float val = r[index];
		r[index] = 1.0 / (1.0 + expf(-val));
	}
}

__global__ void kernelMatrixAbsError(float *p, float *ys, float *r, int rw, int rh)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < rh) && (col < rw)) {
		float pval = p[row * rw + col];
		float ysval = ys[row * rw + col];

		float v = pval - ysval;
		r[row * rw + col] = v * v;
	}
}

__global__ void kernelAbsError(float *p, float *ys, float *r, int m)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m) {
		float pval = p[index];
		float ysval = ys[index];

		float v = pval - ysval;
		r[index] = v * v;
	}
}

__global__ void kernelUpdateParams(float *p, float *ys, float *th, float *xs, int m, float alpha)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m) {
		float h = *p;
		float y = *ys;

		float x = xs[index];

		th[index] = th[index] - alpha * (h - y) * x;
	}
}

__global__ void kernelCrossEntropy(float *p, float *ys, float *r, int m)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m) {
		float pval = p[index];
		float ysval = ys[index];

		float ex = log1pf(expf(-ysval * pval));
		r[index] = ex;
	}
}

__global__ void kernelReduction(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ float partialSum[2 * REDUCE_BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * REDUCE_BLOCK_SIZE;
    if (start + t < len)
       partialSum[t] = input[start + t];
    else
       partialSum[t] = 0;
    if (start + REDUCE_BLOCK_SIZE + t < len)
       partialSum[REDUCE_BLOCK_SIZE + t] = input[start + REDUCE_BLOCK_SIZE + t];
    else
       partialSum[REDUCE_BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = REDUCE_BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }
    // Write the computed sum of the block to the output vector at the correct index
    if (t == 0)
       output[blockIdx.x] = partialSum[0];
}

__global__ void confusion_matrix_kernel(float* d_predictions, float* d_labels, int* d_confusion_matrix, int num_classes, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//int label;
    if (idx < num_samples) {
		int pred = (d_predictions[idx] >= 0.5) ? 1 : 0;;
        int label = static_cast<int>(d_labels[idx]);
       if (pred < num_classes && label < num_classes) {
            atomicAdd(&d_confusion_matrix[label * num_classes + pred], 1);
        }
    }
}

void calculate_confusion_matrix(float* d_predictions, float* d_labels, int* d_confusion_matrix, int num_classes, int num_samples) {
    // Initialize confusion matrix to zero on the GPU
    cudaMemset(d_confusion_matrix, 0, num_classes * num_classes * sizeof(int));
	
    // Launch kernel to compute confusion matrix
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_samples + threadsPerBlock - 1) / threadsPerBlock;

    confusion_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, d_labels, d_confusion_matrix, num_classes, num_samples);

    // Synchronize to ensure all threads are done
    cudaDeviceSynchronize();
}


static void LogisticRegression(Matrix *X, Matrix *y, Matrix *Parameters, Matrix *Train_Parameters, int maxIterations, float alpha, vector<float> &cost_function, int num_classes)
{
	// put data into gpu
	float *gpu_X;
	float *gpu_y;

	float *gpu_prediction;

	float *gpu_params;
	float *gpu_abs_error;
	float *gpu_err_cost;

	float *gpu_predictions;
	int* d_confusion_matrix;
	Matrix predictions;
	AllocateMatrix(&predictions, y->height, y->width);

	Matrix absErrors;
	AllocateMatrix(&absErrors, y->height, y->width);

	float mean_error;
	float sum=0;
	int quantity = 1;

	int m = y->height;

	int numOutputElements;
	numOutputElements = m / (REDUCE_BLOCK_SIZE<<1);
	if (m % (REDUCE_BLOCK_SIZE<<1)) {
		numOutputElements++;
	}

	SAFE_CALL(cudaMalloc((void**)&gpu_X, sizeof(float) * X->width * X->height));
	SAFE_CALL(cudaMalloc((void**)&gpu_y, sizeof(float) * y->width * y->height));
	SAFE_CALL(cudaMalloc((void**)&gpu_prediction, sizeof(float)));
	SAFE_CALL(cudaMalloc((void**)&gpu_predictions, sizeof(float) * y->width * y->height));
	SAFE_CALL(cudaMalloc((void**)&gpu_abs_error, sizeof(float) * y->width * y->height));
	SAFE_CALL(cudaMalloc((void**)&gpu_params, sizeof(float) * Parameters->width * Parameters->height));
	SAFE_CALL(cudaMalloc((void**)&gpu_err_cost, sizeof(float) * numOutputElements));
	SAFE_CALL(cudaMalloc(&d_confusion_matrix, num_classes * num_classes * sizeof(int)));

	SAFE_CALL(cudaMemcpy(gpu_X, X->elements, sizeof(float) * X->width * X->height, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_y, y->elements, sizeof(float) * y->width * y->height, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_params, Parameters->elements, sizeof(float) * Parameters->width * Parameters->height, cudaMemcpyHostToDevice));

	// invoke kernel
	static const int blockWidth = 16;
	static const int blockHeight = blockWidth;
	int numBlocksW = X->width / blockWidth;
	int numBlocksH = X->height / blockHeight;
	if (X->width % blockWidth) numBlocksW++;
	if (X->height % blockHeight) numBlocksH++;

	dim3 dimGrid(numBlocksW, numBlocksH);
	dim3 dimBlock(blockWidth, blockHeight);

	dim3 dimReduce((m - 1) / REDUCE_BLOCK_SIZE + 1);
	dim3 dimReduceBlock(REDUCE_BLOCK_SIZE);

	dim3 dimVectorGrid(((m - 1) / blockWidth * blockWidth) + 1);
	dim3 dimVectorBlock(blockWidth * blockWidth);

	float* error_accum = new float[numOutputElements];
	for (int iter = 0; iter < maxIterations; ++iter) {
		for (int i = 0; i < m; ++i) {
			kernelMatrixMul<<<dimGrid, dimBlock>>>(&gpu_X[i * X->width], gpu_params, gpu_prediction, X->width, Parameters->width, 1, 1);
			kernelSigmoid<<<dimVectorGrid, dimVectorBlock>>>(gpu_prediction, 1);
			kernelUpdateParams<<<dimVectorGrid, dimVectorBlock>>>(gpu_prediction, &gpu_y[i], gpu_params, &gpu_X[i * X->width], Parameters->height, alpha);
		}
		kernelMatrixMul<<<dimGrid, dimBlock>>>(gpu_X, gpu_params, gpu_predictions, X->width, Parameters->width, predictions.width, predictions.height);
		kernelSigmoid<<<dimVectorGrid, dimVectorBlock>>>(gpu_predictions, m);


		// calculate error
		kernelAbsError<<<dimVectorGrid, dimVectorBlock>>>(gpu_predictions, gpu_y, gpu_abs_error, m);
		kernelReduction<<<dimReduce, dimReduceBlock>>>(gpu_abs_error, gpu_err_cost, m);
		SAFE_CALL(cudaMemcpy(error_accum, gpu_err_cost, sizeof(float) * numOutputElements, cudaMemcpyDeviceToHost));
		float g_sum = 0;
		for (int i = 0; i < numOutputElements; ++i)
		{
			g_sum += error_accum[i];
		}

		g_sum /= (2*m);

		cost_function.push_back(g_sum);
		sum += g_sum;
		quantity++;
		cout << g_sum << "\n";
	}

	mean_error = sum/quantity;
	printf("\n The mean error is %f\n", mean_error);

	cout << numOutputElements << "\n";

	int num_samples = y->height;
	cout << num_samples << "\n";
	calculate_confusion_matrix(gpu_predictions, gpu_y, d_confusion_matrix, num_classes, num_samples);

    // Allocate host memory for the confusion matrix and copy result back
    int h_confusion_matrix[4];
    SAFE_CALL(cudaMemcpy(h_confusion_matrix, d_confusion_matrix, num_classes * num_classes * sizeof(int), cudaMemcpyDeviceToHost));
	
	float tp = static_cast< float >(h_confusion_matrix[0]);
	float fp = static_cast< float >(h_confusion_matrix[1]);
	float fn = static_cast< float >(h_confusion_matrix[2]);
	float tn = static_cast< float >(h_confusion_matrix[3]);
	float acc = (tp+tn)/num_samples;
	float f1score = (2*tp)/(2*tp + fp + fn);

	std::cout << "Confusion Matrix:" << std::endl;
    for (int i = 0; i < num_classes; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            std::cout << h_confusion_matrix[i * num_classes + j] << " ";
        }
        std::cout << std::endl;
    }

	std::cout << "Accuracy: " << acc << " ";
	std::cout << "F1 score: " << f1score;

	cout << endl;

	delete[] error_accum;
	SAFE_CALL(cudaFree(gpu_X));
	SAFE_CALL(cudaFree(gpu_y));
	SAFE_CALL(cudaFree(gpu_abs_error));
	SAFE_CALL(cudaFree(gpu_prediction));
	SAFE_CALL(cudaFree(gpu_predictions));
	SAFE_CALL(cudaFree(gpu_params));
	SAFE_CALL(cudaFree(gpu_err_cost));
	SAFE_CALL(cudaFree(d_confusion_matrix));
}

int main(int argc, char *argv[])
{
	string input_file = "";
	int num_classes=2;
	cout << "Please enter a valid file to run test for logistic regression on CUDA:\n>";
	getline(cin, input_file);
 	cout << "You entered: " << input_file << endl << endl;
    Matrix X,y;
    readData (input_file, &X, &y);
    cout <<"\n The X - Matrix." << endl;
    DisplayMatrix (X,true);
    cout <<"\n The y - Matrix." << endl;
    DisplayMatrix (y,true);

    Matrix Parameters, Train_Parameters;
    //Setup matrices with 1 as value initially
    AllocateMatrix(&Parameters, X.width, 1);
    AllocateMatrix(&Train_Parameters, X.width, 1);
    //Initialize with random +1 and -1 parameters.
    InitializeRandom(&Parameters, -1.0, 1.0);

    minMaxNormalization(&X);

    vector<float> cost_function;

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    //unsigned int timer;
    //CUT_SAFE_CALL(cutCreateTimer(&timer));
    
    //cutStartTimer(timer);
    cudaEventRecord(start);
    LogisticRegression(&X, &y, &Parameters, &Train_Parameters, 150, 0.03, cost_function, num_classes);
    //cutStopTimer(timer);
    cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout <<"\n The resulting parameters." << endl;
	DisplayMatrix (Parameters,true);

	//printf("\nProcessing time: %f (ms)\n", cutGetTimerValue(timer));
	printf("\nProcessing time: %f (ms)\n", milliseconds);

    std::cout << "Done" << std::endl;
	system("pause");

	return 0;
}