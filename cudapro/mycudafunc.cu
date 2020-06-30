#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <device_launch_parameters.h>
#include <stdlib.h>

#include <iostream>
#include <memory>
#include <string>

int* pArgc = NULL;
char** pArgv = NULL;
/*序号	名称	值	解释
1	Detected 1 CUDA Capable device(s)	1	检测到1个可用的NVIDIA显卡设备
2	Device 0: "GeForce 930M"	GeForce 930M	当前显卡型号为" GeForce 930M "
3	CUDA Driver Version / Runtime Version	7.5 / 7.5	CUDA驱动版本
4	CUDA Capability Major / Minor version number	5	CUDA设备支持的计算架构版本，即计算能力，该值越大越好

5	Total amount of global memory	4096Mbytes	Global memory全局存储器的大小。使用CUDA RUNTIME API调用函数cudaMalloc后，会消耗GPU设备上的存储空间，合理分配和释放空间避免程序出现crash
6	(3) Multiprocessors, (128) CUDA Cores / MP	384 CUDA Cores	3个流多处理器（即SM），每个多处理器中包含128个流处理器，共384个CUDA核
7	GPU Max Clock rate	941 MHz	GPU最大频率
8	Memory Clock rate	900 MHz	显存的频率
9	Memory Bus Width	64 - bit
10	L2 Cache Size	1048576 bytes
11	Maximum Texture Dimension Size(x, y, z)	1D = (65535)
2D = (65535, 65535)
3D = (4096, 4096, 4096)
12	Maximum Layered 1D Texture Size, (num)layers	1D = (16384), 2048 layers
13	Maximum Layered 2D Texture Size, (num)layers	2D = (16384, 16384), 2048 layers
14	Total amount of constant memory	65535 bytes	常量存储器的大小
15	Total amount of shared memory per block	49152 bytes	共享存储器的大小，共享存储器速度比全局存储器快；多处理器上的所有线程块可以同时共享这些存储器
16	Total number of registers available per block	65535
17	Warp Size	32	Warp，线程束，是SM运行的最基本单位，一个线程束含有32个线程
18	Maximum number of threads per multiprocessor	2048	一个SM中最多有2048个线程，即一个SM中可以有2048 / 32 = 64个线程束Warp
19	Maximum number of threads per block	1024	一个线程块最多可用的线程数目
20	Max dimension size of a thread block(x, y, z)	(1024, 1024, 64)	ThreadIdx.x <= 1024,
ThreadIdx.y <= 1024,
ThreadIdx.z <= 64
Block内三维中各维度的最大值
21	Max dimension size of a grid size (x, y, z) - 21, 474, 836, 476, 553, 500, 000	Grid内三维中各维度的最大值
22	Maximum memory Pitch	2147483647 bytes	显存访问时对齐时的pitch的最大值
23	Texture alignment	512 bytes	纹理单元访问时对其参数的最大值
24	Concurrent copy and kernel execution	Yes with 1 copy engine(s)
25	Run time limit on kernels	Yes
26	Integrated GPU sharing Host Memory	No
27	Support host page - locked memory mapping	Yes
28	Alignment requirement for Surfaces	Yes
29	Device has ECC support	Disabled
30	其他*/

#if CUDART_VERSION < 5000

#include <cuda.h>

template <class T>
inline void getCudaAttribute(T* attribute, CUdevice_attribute device_attribute,
	int device) {
	CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

	if (CUDA_SUCCESS != error) {
		fprintf(
			stderr,
			"cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
			error, __FILE__, __LINE__);

		exit(EXIT_FAILURE);
	}
}

#endif /* CUDART_VERSION < 5000 */

int printdevinfo() {
	//printf("%s Starting...\n\n", argv[0]);
	printf(
		" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
			static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
	}
	else {
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev) {
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
			driverVersion / 1000, (driverVersion % 100) / 10,
			runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
			deviceProp.major, deviceProp.minor);

		char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(msg, sizeof(msg),
			"  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#else
		snprintf(msg, sizeof(msg),
			"  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
#endif
		printf("%s", msg);

		printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
			deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
			deviceProp.multiProcessorCount);
		printf(
			"  GPU Max Clock rate:                            %.0f MHz (%0.2f "
			"GHz)\n",
			deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n",
			deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n",
			deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize) {
			printf("  L2 Cache Size:                                 %d bytes\n",
				deviceProp.l2CacheSize);
		}

#else
		int memoryClock;
		getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
			dev);
		printf("  Memory Clock rate:                             %.0f Mhz\n",
			memoryClock * 1e-3f);
		int memBusWidth;
		getCudaAttribute<int>(&memBusWidth,
			CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
		printf("  Memory Bus Width:                              %d-bit\n",
			memBusWidth);
		int L2CacheSize;
		getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

		if (L2CacheSize) {
			printf("  L2 Cache Size:                                 %d bytes\n",
				L2CacheSize);
		}

#endif

		printf(
			"  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
			"%d), 3D=(%d, %d, %d)\n",
			deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
			deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
			deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf(
			"  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
			deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
		printf(
			"  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
			"layers\n",
			deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
			deviceProp.maxTexture2DLayered[2]);

		printf("  Total amount of constant memory:               %lu bytes\n",
			deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n",
			deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n",
			deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n",
			deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n",
			deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n",
			deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n",
			deviceProp.memPitch);
		printf("  Texture alignment:                             %lu bytes\n",
			deviceProp.textureAlignment);
		printf(
			"  Concurrent copy and kernel execution:          %s with %d copy "
			"engine(s)\n",
			(deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n",
			deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n",
			deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n",
			deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n",
			deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n",
			deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
			deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
			: "WDDM (Windows Display Driver Model)");
#endif
		printf("  Device supports Unified Addressing (UVA):      %s\n",
			deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Device supports Compute Preemption:            %s\n",
			deviceProp.computePreemptionSupported ? "Yes" : "No");
		printf("  Supports Cooperative Kernel Launch:            %s\n",
			deviceProp.cooperativeLaunch ? "Yes" : "No");
		printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
			deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
		printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
			deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

		const char* sComputeMode[] = {
			"Default (multiple host threads can use ::cudaSetDevice() with device "
			"simultaneously)",
			"Exclusive (only one host thread in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this "
			"device)",
			"Exclusive Process (many threads in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Unknown",
			NULL };
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
	}

	if (deviceCount >= 2) {
		cudaDeviceProp prop[64];
		int gpuid[64];  // we want to find the first two GPUs that can support P2P
		int gpu_p2p_count = 0;

		for (int i = 0; i < deviceCount; i++) {
			checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));
			if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
				&& prop[i].tccDriver
#endif
				) {
				// This is an array of P2P capable GPUs
				gpuid[gpu_p2p_count++] = i;
			}
		}

		int can_access_peer;

		if (gpu_p2p_count >= 2) {
			for (int i = 0; i < gpu_p2p_count; i++) {
				for (int j = 0; j < gpu_p2p_count; j++) {
					if (gpuid[i] == gpuid[j]) {
						continue;
					}
					checkCudaErrors(
						cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
					printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
						prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
						can_access_peer ? "Yes" : "No");
				}
			}
		}
	}

	printf("\n");
	std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
	char cTemp[16];

	sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000, (driverVersion % 100) / 10);
#else
	snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
		(driverVersion % 100) / 10);
#endif
	sProfileString += cTemp;

	sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
#else
	snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
		(runtimeVersion % 100) / 10);
#endif
	sProfileString += cTemp;

	sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	sprintf_s(cTemp, 10, "%d", deviceCount);
#else
	snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
	sProfileString += cTemp;
	sProfileString += "\n";
	printf("%s", sProfileString.c_str());
	printf("Result = PASS\n");
	dev = 0;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, dev);
	std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
	std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
	std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
	std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;//1024
	std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
	exit(EXIT_SUCCESS);
}
// 矩阵类型，行优先，M(row, col) = *(M.elements + row * M.width + col)
struct Matrix
{
	int width;
	int height;
	float* elements;
};
// 获取矩阵A的(row, col)元素
__device__ float getElement(Matrix* A, int row, int col)
{
	return A->elements[row * A->width + col];
}

// 为矩阵A的(row, col)元素赋值
__device__ void setElement(Matrix* A, int row, int col, float value)
{
	A->elements[row * A->width + col] = value;
}

// 矩阵相乘kernel，2-D，每个线程计算一个元素
__global__ void matMulKernel(Matrix* A, Matrix* B, Matrix* C)
{
	float Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < A->width; ++i)
	{
		Cvalue += getElement(A, row, i) * getElement(B, i, col);
	}
	setElement(C, row, col, Cvalue);
}

int matrixmul()
{
	int width = 1 << 10;
	int height = 1 << 10;
	Matrix* A, * B, * C;
	// 申请托管内存
	cudaMallocManaged((void**)&A, sizeof(Matrix));
	cudaMallocManaged((void**)&B, sizeof(Matrix));
	cudaMallocManaged((void**)&C, sizeof(Matrix));
	int nBytes = width * height * sizeof(float);
	cudaMallocManaged((void**)&A->elements, nBytes);
	cudaMallocManaged((void**)&B->elements, nBytes);
	cudaMallocManaged((void**)&C->elements, nBytes);

	// 初始化数据
	A->height = height;
	A->width = width;
	B->height = height;
	B->width = width;
	C->height = height;
	C->width = width;
	for (int i = 0; i < width * height; ++i)
	{
		A->elements[i] = 1.0;
		B->elements[i] = 2.0;
	}

	// 定义kernel的执行配置
	dim3 blockSize(16, 16);//比较得到256比较快
	//std::cout << (width) / blockSize.x;//(width) / blockSize.x==(width + blockSize.x - 1) / blockSize.x
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);
	// 执行kernel
	matMulKernel << < gridSize, blockSize >> > (A, B, C);


	// 同步device 保证结果能正确访问
	cudaDeviceSynchronize();
	// 检查执行结果
	float maxError = 0.0;
	for (int i = 0; i < width * height; ++i)
		maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
	std::cout << "最大误差: " << maxError << std::endl;

	return 0;
}


__global__ void add(float* x, float* y, float* z, int n)
{
	// 获取全局索引
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	// 步长
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
	{
		z[i] = x[i] + y[i];
	}
}
int matrixadd()
{
	int N = 1 << 20;
	int nBytes = N * sizeof(float);
	// 申请host内存
	float* x, * y, * z;
	x = (float*)malloc(nBytes);
	y = (float*)malloc(nBytes);
	z = (float*)malloc(nBytes);

	// 初始化数据
	for (int i = 0; i < N; ++i)
	{
		x[i] = 10.0;
		y[i] = 20.0;
	}

	// 申请device内存
	float* d_x, * d_y, * d_z;
	cudaMalloc((void**)&d_x, nBytes);
	cudaMalloc((void**)&d_y, nBytes);
	cudaMalloc((void**)&d_z, nBytes);

	// 将host数据拷贝到device
	cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
	// 定义kernel的执行配置
	dim3 blockSize(256);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
	// 执行kernel
	add << < gridSize, blockSize >> > (d_x, d_y, d_z, N);

	// 将device得到的结果拷贝到host
	cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyHostToDevice);

	// 检查执行结果
	float maxError = 0.0;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(z[i] - 30.0));
	std::cout << "最大误差: " << maxError << std::endl;

	// 释放device内存
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	// 释放host内存
	free(x);
	free(y);
	free(z);

	return 0;
}
