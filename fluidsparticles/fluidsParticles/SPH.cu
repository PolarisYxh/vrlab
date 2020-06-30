#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <helper_math.h>
#include "global.h"
#include "SPH.h"

extern struct log frame_log;

__global__ void init_fluid_CUDA(float3 *pos, float *mass, int *num, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int x = i/(32*32);
	int y = (i-x*32*32)/32;
	int z = i-(x*32+y)*32;
	int _i;
	if((x-15.5f)*(x-15.5f)+(y-15.5f)*(y-15.5f)+(z-15.5f)*(z-15.5f)<16.0f*16.0f)
	{
		_i = atomicAdd(num,1);
		pos[_i] = (make_float3(make_int3(x,y,z)) + make_float3(9.5f, 3.0f, 9.5f))*para->sph_spacing;
		mass[_i] = para->sph_m0;
	}
	return ;
}

__global__ void init_fluid_CUDA(float3 *pos, float *mass, int num, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	int x = i/(25*25);
	int y = (i-x*25*25)/25;
	int z = i-(x*25+y)*25;
	pos[i] = (make_float3(0.5f, 0.0f, 0.0f)*(y%2) + make_float3(0.0f, 0.5f, 0.0f)*(z%2)
		+ make_float3(0.0f, 0.0f, 0.5f)*(x%2) + make_float3(make_int3(x, y, z))
		+ make_float3(13.0f, 3.0f, 13.0f))*para->sph_spacing;
 	mass[i] = para->sph_m0;
// 	if(i==0)
// 	{
// 		pos[0] = (make_float3(0.1f, 0.0f, 0.0f) + make_float3(25.0f, 25.0f, 25.0f))*para->sph_spacing;
// 		pos[1] = (make_float3(1.0f, 0.1f, 0.0f) + make_float3(25.0f, 25.0f, 25.0f))*para->sph_spacing;
// 		pos[2] = (make_float3(0.0f, 1.1f, 0.0f) + make_float3(25.0f, 25.0f, 25.0f))*para->sph_spacing;
// 		pos[3] = (make_float3(1.1f, 1.0f, 0.0f) + make_float3(25.0f, 25.0f, 25.0f))*para->sph_spacing;
// 		pos[4] = (make_float3(0.0f, 0.0f, 1.0f) + make_float3(25.0f, 25.0f, 25.0f))*para->sph_spacing;
// 		pos[5] = (make_float3(1.0f, 0.0f, 1.0f) + make_float3(25.0f, 25.0f, 25.0f))*para->sph_spacing;
// 		pos[6] = (make_float3(0.0f, 1.0f, 1.0f) + make_float3(25.0f, 25.0f, 25.0f))*para->sph_spacing;
// 		pos[7] = (make_float3(1.0f, 1.0f, 1.1f) + make_float3(25.0f, 25.0f, 25.0f))*para->sph_spacing;
// 	}
	return ;
}

__global__ void init_boundary_CUDA(float3 *pos, float *mass, int num, int3 cellSize, float3 spaceSize, int3 compactSize)
{
	int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	int3 p;
	int start=0;
	if(i>=0 && i<compactSize.x*compactSize.y)	// back
	{
		p.x = i/compactSize.y;
		p.y = i-p.x*compactSize.y;
		p.z = 0;
		pos[start+i] = make_float3(p)/make_float3(compactSize-make_int3(1))*spaceSize;
	}
	i -= compactSize.x*compactSize.y;
	start += compactSize.x*compactSize.y;
	if(i>=0 && i<compactSize.x*compactSize.y)	// front
	{
		p.x = i/compactSize.y;
		p.y = i-p.x*compactSize.y;
		p.z = compactSize.z-1;
		pos[start+i] = make_float3(p) / make_float3(compactSize - make_int3(1))*spaceSize;
	}
	i -= compactSize.x*compactSize.y;
	start += compactSize.x*compactSize.y;
	if(i>=0 && i<compactSize.x*(compactSize.z-2))	// bottom
	{
		p.x = i/(compactSize.z-2);
		p.y = 0;
		p.z = 1+i-p.x*(compactSize.z-2);
		pos[start+i] = make_float3(p) / make_float3(compactSize - make_int3(1))*spaceSize;
	}
	i -= compactSize.x*(compactSize.z-2);
	start += compactSize.x*(compactSize.z-2);
	if(i>=0 && i<compactSize.x*(compactSize.z-2))	// top
	{
		p.x = i/(compactSize.z-2);
		p.y = compactSize.y-1;
		p.z = 1+i-p.x*(compactSize.z-2);
		pos[start+i] = make_float3(p) / make_float3(compactSize - make_int3(1))*spaceSize;
	}
	i -= compactSize.x*(compactSize.z-2);
	start += compactSize.x*(compactSize.z-2);
	if(i>=0 && i<(compactSize.y-2)*(compactSize.z-2))	// left
	{
		p.x = 0;
		p.y = i/(compactSize.z-2);
		p.z = i-p.y*(compactSize.z-2);
		p+=make_int3(0,1,1);
		pos[start+i] = make_float3(p) / make_float3(compactSize - make_int3(1))*spaceSize;
	}
	i -= (compactSize.y-2)*(compactSize.z-2);
	start += (compactSize.y-2)*(compactSize.z-2);
	if(i>=0 && i<(compactSize.y-2)*(compactSize.z-2))	// right
	{
		p.x = compactSize.x-1;
		p.y = i/(compactSize.z-2);
		p.z = i-p.y*(compactSize.z-2);
		p+=make_int3(0,1,1);
		pos[start+i] = make_float3(p) / make_float3(compactSize - make_int3(1))*spaceSize;
	}
	i+=start;
	pos[i] = 0.99f*pos[i] + 0.005f*spaceSize;//乘0.99避免边界上粒子被误判为cell之外
	return;
}

inline __device__ int particlePos2cellIdx(int3 pos, int3 cellSize)
{	
	// return (cellSize.x*cellSize.y*cellSize.z) if the particle is out of the grid
	return (pos.x>=0 && pos.x<cellSize.x && pos.y>=0 && pos.y<cellSize.y && pos.z>=0 && pos.z<cellSize.z)? 
		(((pos.x*cellSize.y)+pos.y)*cellSize.z+pos.z) 
		: (cellSize.x*cellSize.y*cellSize.z);
}

__global__ void mapParticles2cells_CUDA(int *particles2cells, int3 cellSize, float3 *pos, int num, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	particles2cells[i] = particlePos2cellIdx(make_int3(pos[i]/para->sph_cell_length), cellSize);
	return ;
}

extern "C" __global__ void countingInCell_CUDA(int* cellStart, int *particles2cells, int num)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	atomicAdd(&cellStart[particles2cells[i]],1);
	return ;
}

inline __device__ float cubic_spline_kernel(float r, SPH_ParameterSet *para)
{
	float q = 2.0f*fabs(r)/para->sph_smoothing_radius;
	return (q<para->sph_epsilon)?0.0f:
		((q)<=1.0f?(pow(2.0f-q,3)-4.0f*pow(1.0f-q,3)):
			(q)<=2.0f?(pow(2.0f-q,3)):
			0.0f) / (4.0f*M_PI*pow(para->sph_smoothing_radius, 3));
}

inline __device__ float3 cubic_spline_kernel_gradient(float3 r, SPH_ParameterSet *para)
{
	float q = 2.0f*length(r)/para->sph_smoothing_radius;
	return 
			((q) <= 1.0f ? -(3.0f*(2.0f-q)*(2.0f-q)-12.0f*(1.0f-q)*(1.0f-q)) :
			 (q) <= 2.0f ? -(3.0f*(2.0f-q)*(2.0f-q)) :
			 0.0f) / (2.0f*M_PI*pow(para->sph_smoothing_radius, 4)) * r / fmaxf(para->sph_epsilon, length(r));
}

inline __device__ float poly6_kernel(float r2, SPH_ParameterSet *para)
{	
	return (r2<=para->sph_smoothing_radius*para->sph_smoothing_radius) ? 315.0f*pow((para->sph_smoothing_radius*para->sph_smoothing_radius-r2),3)/(64.0f*M_PI*pow(para->sph_smoothing_radius,9)) :0.0f;
}

inline __device__ float3 poly6_kernel_gradient(float3 r, SPH_ParameterSet *para)
{	
	return (length(r)<=para->sph_smoothing_radius) ? (r)*(-945.0f*pow((para->sph_smoothing_radius*para->sph_smoothing_radius-dot(r,r)),2)/(32.0f*M_PI*pow(para->sph_smoothing_radius,9))) : make_float3(0.0f); //约掉了length(r)，不然会除0
}

inline __device__ float3 surface_tension_kernel_gradient(float3 r, SPH_ParameterSet *para)	
{
/************************************************************************/
/*  
black magic!                                                        
一个自定义的kernel曲线，专门处理[2014][TOG][Robust Simulation of Small-Scale Thin Features in SPH-based Free Surface Flows]中提到的surface tension                                                      
自变量q=length(r)/h,定义域[0,1]，由两段piece-wise的六次曲线构成([0,0.5]和[0.5,2])，在0处一阶导数为0，在1处值为0，一阶导数为0，在0.5处取最值，一阶导数为0，二阶导数连续，在2处一阶导数为0，值为0，在[0,0.5]区间内穿过x轴(即粒子相互靠太近时吸引变排斥)
该核函数考虑到引力稳定性，经过特殊设计，最重要的一个特点为：从0.5向0变化和从0.5向1变化同样的距离相比，从0.5向1减小的更慢，这样远离时引力减小更慢，靠近时引力减小更快，可以保持引力的稳定（参见文中Fig.5(a))
该kernel曲线已经过三维球体的归一化
                                                                        */
/************************************************************************/
// 	float q = length(r)/para->sph_smoothing_radius;
// 	return
// 		(q<para->sph_epsilon) ? make_float3(0.0f):
// 		((q) <= 0.5f ? (((((16.0f*q-21.0f)*q+8.25f)*q-1.0f)*q+0.1875f)*q*q-0.0156f):
// 		 (q) <= 1.0f ?  pow(1.0f-q,3)*q*q*q:
// 		0.0f) / (0.007291667f*M_PI*pow(para->sph_smoothing_radius, 3)) * -r / fmaxf(para->sph_epsilon, length(r));

/************************************************************************/
/*
[2013][SIGGRAPH ASIA][Versatile Surface Tension and Adhesion for SPH Fluids]中的核函数，比自定义核更稳定收敛更快
该kernel曲线已经过三维球体的归一化
*/
/************************************************************************/
	float x=length(r);
	return
		(x<para->sph_epsilon) ? make_float3(0.0f):(
		2.0f*x <= para->sph_smoothing_radius ? 2.0f*pow((para->sph_smoothing_radius-x),3)*x*x*x-0.0156f*pow(para->sph_smoothing_radius,6):
		x      <= para->sph_smoothing_radius ?      pow((para->sph_smoothing_radius-x),3)*x*x*x:
		0.0f) * 136.0241f/ (M_PI*pow(para->sph_smoothing_radius, 9)) * -r / fmaxf(para->sph_epsilon, x);
}

inline __device__ float poly6_kernel_laplacian(float r2, SPH_ParameterSet *para)
{	
	return (r2<=para->sph_smoothing_radius*para->sph_smoothing_radius) ? ((945.0f*r2*(para->sph_smoothing_radius*para->sph_smoothing_radius - r2))/(8.0f*M_PI*pow(para->sph_smoothing_radius,9)) - (945.0f*pow((para->sph_smoothing_radius*para->sph_smoothing_radius - r2),2))/32.0f*M_PI*pow(para->sph_smoothing_radius,9)):0.0f;
}

inline __device__ float viscosity_kernel_laplacian(float r, SPH_ParameterSet *para) {
	return (r<=para->sph_smoothing_radius) ? (45.0f*(para->sph_smoothing_radius-r)/(M_PI*pow(para->sph_smoothing_radius, 6))):0.0f;
}

__device__ void contributeBoundaryParticlesKernel(float *sum_kernel, int i, int cellID, float3 *pos, int *cellStart, int3 cellSize, SPH_ParameterSet *para)
{
	int j,end;
	if(cellID==(cellSize.x*cellSize.y*cellSize.z)) return;
	j = cellStart[cellID];	end = cellStart[cellID+1];
	while(j<end)
	{
		*sum_kernel += cubic_spline_kernel(length(pos[i]-pos[j]), para);
		j++;
	}
	return;
}

__global__ void computeBoundaryParticlesMass(float3 *pos, float *mass, int particleNum, int *cellStart, int3 cellSize, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=particleNum) return;
	int3 cellPos = make_int3(pos[i]/para->sph_cell_length);
	int cellID;
#pragma unroll
	for(int m=0; m<27; m++)
	{
		cellID = particlePos2cellIdx(cellPos+make_int3(m/9-1,(m%9)/3-1,m%3-1), cellSize);	
		contributeBoundaryParticlesKernel(&mass[i], i, cellID, pos, cellStart, cellSize, para);
	}
	mass[i] = para->sph_rhob / fmaxf(para->sph_epsilon, mass[i]);
	return ;
}

void SPH::init_fluid()
{
	CUDA_CALL(cudaMemset(pos_fluid, 0, sizeof(float3)*num_fluid));
	CUDA_CALL(cudaMemset(vel_fluid, 0, sizeof(float3)*num_fluid));
	CUDA_CALL(cudaMemset(mass_fluid, 0, sizeof(float)*num_fluid));
	CUDA_CALL(cudaMemset(pos_staticBoundary, 0, sizeof(float3)*num_staticBoundary));
	CUDA_CALL(cudaMemset(mass_staticBoundary, 0, sizeof(float)*num_staticBoundary));

	CUDA_CALL(cudaMemset(density, 0, sizeof(float)*num_fluid));
	CUDA_CALL(cudaMemset(pressure, 0, sizeof(float)*num_fluid));
	CUDA_CALL(cudaMemset(particles2cells_fluid, 0, sizeof(int)*num_fluid));
	CUDA_CALL(cudaMemset(particles2cells_staticBoundary, 0, sizeof(int)*num_fluid));
	CUDA_CALL(cudaMemset(cellStart_fluid, 0, sizeof(int)*(cellSize.x*cellSize.y*cellSize.z + 1)));
	CUDA_CALL(cudaMemset(cellStart_staticBoundary, 0, sizeof(int)*(cellSize.x*cellSize.y*cellSize.z + 1)));

	CUDA_CALL(cudaMemset(color_grad, 0, sizeof(float3)*num_fluid));
	CUDA_CALL(cudaMemset(alpha, 0, sizeof(float)*num_fluid));
	CUDA_CALL(cudaMemset(error, 0, sizeof(float)*num_fluid));

	CUDA_CALL(cudaMemset(stiff_den, 0, sizeof(float)*num_fluid));
	CUDA_CALL(cudaMemset(stiff_div, 0, sizeof(float)*num_fluid));

	CUDA_CALL(cudaMemset(mean_pos, 0, sizeof(float3)*num_fluid));
	CUDA_CALL(cudaMemset(covariance, 0, sizeof(square3)*num_fluid));
	CUDA_CALL(cudaMemset(singular, 0, sizeof(float3)*num_fluid));
	CUDA_CALL(cudaMemset(probableSurface, 0, sizeof(float)*num_fluid));


	//初始化流体粒子
	int gridsize = (num_fluid-1)/BLOCK_SIZE+1;

// 	int *cnt, cnt_h;
// 	cudaMalloc((void**)&cnt, sizeof(int));
// 	cudaMemset(cnt, 0, sizeof(int));
// 	init_fluid_CUDA<<<32*32*32/BLOCK_SIZE, BLOCK_SIZE>>>(pos_fluid, mass_fluid, cnt, para);
// 	cudaDeviceSynchronize(); CHECK_KERNEL();
// 	cudaMemcpy(&cnt_h, cnt, sizeof(int),cudaMemcpyDeviceToHost);
// 	printf("%d fluid paritles initiated.\n", cnt_h);

// 	init_fluid_CUDA<<<gridsize, BLOCK_SIZE>>>(pos_fluid, mass_fluid, num_fluid, para);
// 	cudaDeviceSynchronize(); CHECK_KERNEL();

	int a=32, b=20;
	num_fluid = a*b*b;
	float3 *fluid_h = (float3*)malloc(sizeof(float3)*num_fluid);
	for (int i=0; i<a; i++)
		for (int j=0; j<b; j++)
			for (int k=0; k<b; k++)
				fluid_h[(i*b+j)*b+k] = make_float3(0.16f+para_h->sph_spacing*(float)j, 0.1f+para_h->sph_spacing*(float)i, 0.08f+para_h->sph_spacing*(float)k)
				- 0.166666666665f*1.001f*para_h->sph_spacing; 
	cudaMemcpy(pos_fluid, fluid_h, sizeof(float3)*num_fluid, cudaMemcpyHostToDevice);
	free(fluid_h);

// 	int a=8, b=48;
// 	num_fluid = a*b*b;
// 	float3 *fluid_h = (float3*)malloc(sizeof(float3)*num_fluid);
// 	for (int i=0; i<a; i++)
// 		for (int j=0; j<b; j++)
// 			for (int k=0; k<b; k++)
// 				fluid_h[(i*b+j)*b+k] = make_float3(0.0275f+para_h->sph_spacing*(float)j, 0.025f+para_h->sph_spacing*(float)i, 0.0275f+para_h->sph_spacing*(float)k); // Trick:让初始状态与ghost粒子交错，看起来效果好些
// 	float radius = 5.2f;
// 	int radius_d = ceil(radius);
// 	int sphere_count = 0;
// 	float3 *sphere_h = (float3*)malloc(sizeof(float3)*8*radius_d*radius_d*radius_d);
// 	for (int i=-radius_d; i<=radius_d; i++)
// 		for (int j=-radius_d; j<=radius_d; j++)
// 			for (int k=-radius_d; k<=radius_d; k++) {
// 				float3 p = make_float3(make_int3(i, j, k));
// 				if (length(p)<radius) {
// 					sphere_h[sphere_count++] = p*para_h->sph_spacing + make_float3(0.5f, 0.5f, 0.5f);
// 				}
// 			}
// 
// 	cudaMemcpy(pos_fluid, fluid_h, sizeof(float3)*num_fluid, cudaMemcpyHostToDevice);
// 	cudaMemcpy(pos_fluid+num_fluid, sphere_h, sizeof(float3)*sphere_count, cudaMemcpyHostToDevice);
// 	thrust::fill(thrust::device, vel_fluid, vel_fluid+num_fluid, make_float3(0.0f, 0.0f, 0.0f));
// 	thrust::fill(thrust::device, vel_fluid+num_fluid, vel_fluid+num_fluid+sphere_count, make_float3(0.0f, -8.0, 0.0f));
// 	free(fluid_h);
// 	free(sphere_h);
// 	num_fluid+=sphere_count;


	thrust::fill(thrust::device, mass_fluid, mass_fluid+num_fluid, para_h->sph_m0);

	//初始化static边界粒子的位置
	gridsize = (num_staticBoundary-1)/BLOCK_SIZE+1;
	init_boundary_CUDA<<<gridsize, BLOCK_SIZE>>>(pos_staticBoundary, mass_staticBoundary, num_staticBoundary, cellSize, spaceSize, compactSize);
	cudaDeviceSynchronize(); CHECK_KERNEL();
	//在static边界粒子上建立SPH的结构，只需要建立一次，以后不用在每个时间步里更新
	mapParticles2cells_CUDA<<<gridsize, BLOCK_SIZE>>>(particles2cells_staticBoundary, cellSize, pos_staticBoundary, num_staticBoundary, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();
	thrust::sort_by_key(thrust::device, particles2cells_staticBoundary, particles2cells_staticBoundary+num_staticBoundary, pos_staticBoundary);
	countingInCell_CUDA<<<gridsize, BLOCK_SIZE>>>(cellStart_staticBoundary, particles2cells_staticBoundary, num_staticBoundary);
	cudaDeviceSynchronize(); CHECK_KERNEL();
	thrust::exclusive_scan(thrust::device, cellStart_staticBoundary, cellStart_staticBoundary+cellSize.x*cellSize.y*cellSize.z+1, cellStart_staticBoundary);
	//求每个static边界粒子的质量
	computeBoundaryParticlesMass<<<gridsize, BLOCK_SIZE>>>(pos_staticBoundary, mass_staticBoundary, num_staticBoundary, cellStart_staticBoundary, cellSize, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();

	neighborSearch();
#ifdef DFSPH
	computeDensityAlpha();
	surfaceDetection();
#else
	computeDensity();
#endif
	return;
}

void SPH::neighborSearch()
{
	int gridsize = (num_fluid-1)/BLOCK_SIZE+1;
	mapParticles2cells_CUDA<<<gridsize, BLOCK_SIZE>>>(particles2cells_fluid, cellSize, pos_fluid, num_fluid, para);
	cudaMemcpy(particles2cells2_fluid, particles2cells_fluid, sizeof(int)*num_fluid, cudaMemcpyDeviceToDevice);
	thrust::sort_by_key(thrust::device, particles2cells_fluid, particles2cells_fluid + num_fluid, pos_fluid);
	cudaMemcpy(particles2cells_fluid, particles2cells2_fluid, sizeof(int)*num_fluid, cudaMemcpyDeviceToDevice);
	thrust::sort_by_key(thrust::device, particles2cells_fluid, particles2cells_fluid + num_fluid, vel_fluid);

	cudaMemset(cellStart_fluid, 0, sizeof(int)*(cellSize.x*cellSize.y*cellSize.z+1));
	countingInCell_CUDA<<<gridsize, BLOCK_SIZE>>>(cellStart_fluid, particles2cells_fluid, num_fluid);
	thrust::exclusive_scan(thrust::device, cellStart_fluid, cellStart_fluid+cellSize.x*cellSize.y*cellSize.z+1, cellStart_fluid);
	return;
}

#ifndef DFSPH
__device__ void contributeFluidDensity(float *density, int i, float3 *pos, float *mass, int cellStart, int cellEnd, SPH_ParameterSet *para)
{
	int j = cellStart;
	while(j<cellEnd)
	{
		*density += mass[j] * cubic_spline_kernel(length(pos[i]-pos[j]), para);
		j++;
	}
	return;
}

__device__ void contributeBoundaryDensity(float *density, float3 *pos_i, float3 *pos, float *mass, int cellStart, int cellEnd, SPH_ParameterSet *para)
{
	int j = cellStart;
	while(j<cellEnd)
	{
		*density += mass[j] * cubic_spline_kernel(length(*pos_i - pos[j]), para);
		j++;
	}
	return;
}

__global__ void computeDensity_CUDA(float *density, int num, float3 *pos_fluid, float *mass_fluid, int *cellStart_fluid, int3 cellSize,
									float3 *pos_staticBoundary, float *mass_staticBoundary, int *cellStart_staticBoundary, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	int cellID;
	__syncthreads();

	for(int m=0; m<27; __syncthreads(),m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_fluid[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributeFluidDensity(&density[i], i, pos_fluid, mass_fluid, cellStart_fluid[cellID], cellStart_fluid[cellID+1], para);
		contributeBoundaryDensity(&density[i], &pos_fluid[i], pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary[cellID], cellStart_staticBoundary[cellID+1], para);
	}
	return ;
}

__global__ void computePressure_CUDA(float *pressure, float *density, int num, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	//pressure[i] = para->sph_k*(density[i] - para->sph_rho0);
	pressure[i] = para->sph_k* (pow((density[i]/para->sph_rho0),7)-1.0f);
	//clamp
	if(pressure[i]<0.0f) pressure[i]=0.0f;
	return;
}

void SPH::computeDensity()
{
	int gridsize = (num_fluid-1)/BLOCK_SIZE+1;
	CUDA_CALL(cudaMemset(density, 0, sizeof(float)*num_fluid));
	computeDensity_CUDA <<<gridsize, BLOCK_SIZE>>>(density, num_fluid, pos_fluid, mass_fluid, cellStart_fluid, cellSize, pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();
	computePressure_CUDA <<<gridsize, BLOCK_SIZE>>>(pressure, density, num_fluid, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();

    static float *density_h  = (float*)malloc(sizeof(float)*num_fluid);
    cudaMemcpy(density_h, density, sizeof(float)*num_fluid, cudaMemcpyDeviceToHost);
    float *max_den = thrust::max_element(density_h, density_h+num_fluid);
	float *min_den = thrust::min_element(density_h, density_h+num_fluid);
    printf("max density = %.12f\t min density = %.12f\n", *max_den, *min_den); 

	return;
}

__device__ void contributeFluidPressure(float3 *a, int i, float3 *pos, float *mass, float *density, float *pressure, int cellStart, int cellEnd, SPH_ParameterSet *para)
{
	int j = cellStart;
	while(j<cellEnd)
	{
		if(i!=j)
			*a += -mass[j]*(pressure[i]/fmaxf(para->sph_epsilon, density[i]*density[i])+pressure[j]/fmaxf(para->sph_epsilon, density[j]*density[j]))*cubic_spline_kernel_gradient(pos[i]-pos[j], para);
		j++;
	}
	return;
}

__device__ void contributeBoundaryPressure(float3 *a, float3 pos_i, float3 *pos, float *mass, float density, float pressure, int cellStart, int cellEnd, SPH_ParameterSet *para)
{
	int j = cellStart;
	while(j<cellEnd)
	{
		*a += -mass[j]*(pressure/fmaxf(para->sph_epsilon, density*density))*cubic_spline_kernel_gradient(pos_i-pos[j], para);
		j++;
	}
	return;
}

__global__ void pressureForce_CUDA(float3 *vel_fluid, float3 *pos_fluid, float *mass_fluid, float *density, float *pressure, int num, int *cellStart_fluid, int3 cellSize,
								   float3 *pos_staticBoundary, float *mass_staticBoundary, int *cellStart_staticBoundary, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	float3 a=make_float3(0.0f);
	int cellID;
	__syncthreads();

	for(int m=0; m<27; __syncthreads(),m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_fluid[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributeFluidPressure(&a, i, pos_fluid, mass_fluid, density, pressure, cellStart_fluid[cellID], cellStart_fluid[cellID+1], para);
		contributeBoundaryPressure(&a, pos_fluid[i], pos_staticBoundary, mass_staticBoundary, density[i], pressure[i], cellStart_staticBoundary[cellID], cellStart_staticBoundary[cellID+1], para);
	}

	if (length(a)>1000.0f)
		a = normalize(a)*1000.0f;

	vel_fluid[i] += a*para->sph_dt;
	return ;
}
#endif

__global__ void externalForce_CUDA(float3 *vel, int num, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	vel[i] += para->sph_dt*para->sph_g;
	return;
}

__device__ void contributeXsphViscosity(float3 *a, int i, float3 *pos, float3 *vel, float *mass, float *density, int j/*cellStart*/, int cellEnd, SPH_ParameterSet *para)
{
	while(j<cellEnd)
	{
		*a += para->xsph_e* mass[j]*((vel[j]-vel[i])/para->sph_rho0)*cubic_spline_kernel(length(pos[i]-pos[j]), para);
		j++;
	}
	return;
}

__global__ void xsphViscosity_CUDA(float3 *vel, float3 *pos, float *mass, float *density, int num, int *cellStart, int3 cellSize, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	float3 a=make_float3(0.0f);
	int cellID;
	__syncthreads();
#pragma unroll
	for(int m=0; m<27; __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributeXsphViscosity(&a, i, pos, vel, mass, density, cellStart[cellID], cellStart[cellID+1], para);		
	}

	vel[i] += a;
	return ;
}

__device__ void contributeViscosity(float3 *a, int i, float3 *pos, float3 *vel, float *mass, float *density, int j/*cellStart*/, int cellEnd, SPH_ParameterSet *para) {
	while (j<cellEnd) {
		*a +=  mass[j]*((vel[j]-vel[i])/para->sph_rho0)*viscosity_kernel_laplacian(length(pos[i]-pos[j]), para);
		j++;
	}
	return;
}

__global__ void viscosity_CUDA(float3 *force, float3 *vel, float3 *pos, float *mass, float *density, int num, int *cellStart, int3 cellSize, SPH_ParameterSet *para) {
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	float3 a=make_float3(0.0f);
	int cellID;
	__syncthreads();
#pragma unroll
	for (int m=0; m<27; __syncthreads(), m++) {
		cellID = particlePos2cellIdx(make_int3(pos[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if (cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributeViscosity(&a, i, pos, vel, mass, density, cellStart[cellID], cellStart[cellID+1], para);
	}

	force[i] = para->visc*a*para->sph_dt;
	return;
}

// __device__ void contributeBoundaryFriction(float3 *a, float3 pos_i, float3 vel_i, int cellID, Particle *particles, float density, int *cellStart, int3 cellSize)
// {
// 	int j,end;
// 	if(cellID==(cellSize.x*cellSize.y*cellSize.z)) return;
// 	j = cellStart[cellID];	end = cellStart[cellID+1];
// 	while(j<end)
// 	{
// 		*a += -particles[j].m*(
// 			sph_boundary_friction*sph_smoothing_radius/(2*density)*
// 			max(dot(vel_i-particles[j].v,pos_i-particles[j].x),0.0)/
// 			(dot(pos_i-particles[j].x,pos_i-particles[j].x)+0.01*sph_smoothing_radius*sph_smoothing_radius)
// 			)*poly6_kernel_gradient(pos_i-particles[j].x, sph_smoothing_radius);
// 		j++;
// 	}
// 	return;
// }
// 
// __global__ void boundaryFriction_CUDA(Particle *fluidParticles, float *density, int fluidParticleNum, int *cellStart_fluid, int3 cellSize,
// 								   Particle*staticBoundaryParticles, int *cellStart_staticBoundary)
// {
// 	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
// 	if (i>=fluidParticleNum) return;
// 	float3 a=make_float3(0.0);
// 	int3 cellPos = make_int3(fluidParticles[i].x/sph_spacing);
// 	int cellID;
// 
// 	for(int m=0; m<27; m++)
// 	{
// 		cellID = particlePos2cellIdx(cellPos+make_int3(m/9-1,(m%9)/3-1,m%3-1), cellSize);	
// 		contributeBoundaryFriction(&a, fluidParticles[i].x, fluidParticles[i].v, cellID, staticBoundaryParticles, density[i], cellStart_staticBoundary, cellSize);
// 	}
// 
// 	fluidParticles[i].v += a*sph_dt;
// 	return ;
// }


__device__ void contributeSurfaceTensionAndAirPressure(float3 *a, int i, int cellID, float3 *pos, float *density, float *mass, float3* color_grad, int j, int cellEnd, SPH_ParameterSet *para)
{
	while(j<cellEnd)
	{
		// surface tension
// 		*a += 0.25f*mass[j]/(para->sph_rho0*para->sph_rho0)*para->sph_color_energy_coefficient*
// 			(dot(color_grad[i], color_grad[i])+dot(color_grad[j], color_grad[j]))
// 			* surface_tension_kernel_gradient(pos[i]-pos[j], para);
		// air pressure
		*a += para->sph_p_atm *mass[j]/(para->sph_rho0*para->sph_rho0)*cubic_spline_kernel_gradient(pos[i]-pos[j], para) /*following terms disable inner particles*/* length(color_grad[i])/fmaxf(para->sph_epsilon, length(color_grad[i]));
		j++;
	}
	return;
}

__global__ void surfaceTensionAndAirPressure_CUDA(float3 *vel, float3 *pos_fluid, float *density, float *mass_fluid, float3 *color_grad, int num, int *cellStart, int3 cellSize, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	float3 a=make_float3(0.0f);
	int cellID;
#pragma unroll
	for(int m=0; m<27; __syncthreads(),m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_fluid[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributeSurfaceTensionAndAirPressure(&a, i, cellID, pos_fluid, density, mass_fluid, color_grad, cellStart[cellID], cellStart[cellID+1], para);
	}
	vel[i] += a*para->sph_dt;
	return ;
}

#ifndef DFSPH
void SPH::applyForce()
{
	int gridsize = (num_fluid-1)/BLOCK_SIZE+1;
	externalForce_CUDA<<<gridsize, BLOCK_SIZE>>>(vel_fluid, num_fluid, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();
// 	// 采用XSPH来仿真黏性 [Ghost SPH for Animating Water]
// 	xsphViscosity_CUDA<<<gridsize, BLOCK_SIZE>>>(vel_fluid, pos_fluid, mass_fluid, density, num_fluid, cellStart_fluid, cellSize, para);
	viscosity_CUDA<<<gridsize, BLOCK_SIZE>>>(buffer_float3, vel_fluid, pos_fluid, mass_fluid, density, num_fluid, cellStart_fluid, cellSize, para);
 	cudaDeviceSynchronize(); CHECK_KERNEL();
	thrust::transform(thrust::device, vel_fluid, vel_fluid+num_fluid, buffer_float3, vel_fluid, thrust::plus<float3>());

	// 边界的处理方法参考[siggraph12][Versatile Rigid-Fluid Coupling for Incompressible SPH]
	// 其实如果不是双向耦合的话，从边界到流体的摩擦力并不能很明显的观察到
	//boundaryFriction_CUDA<<<gridsize, BLOCK_SIZE>>>(fluidParticles, density, fluidParticleNum, cellStart_fluid, cellSize, staticBoundaryParticles, cellStart_staticBoundary);
	//cudaDeviceSynchronize(); CHECK_KERNEL();

	// 表面张力和空气压力的处理方法参考[tog14][Robust Simulation of Small-Scale Thin Features in SPH-based Free Surface Flows]
	surfaceDetection();
	surfaceTensionAndAirPressure_CUDA<<<gridsize, BLOCK_SIZE>>>(vel_fluid, pos_fluid, density, mass_fluid, color_grad, num_fluid, cellStart_fluid, cellSize, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();

	pressureForce_CUDA<<<gridsize, BLOCK_SIZE>>>(vel_fluid, pos_fluid, mass_fluid, density, pressure, num_fluid, cellStart_fluid, cellSize, pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();
	return;
}
#endif

__global__ void moveParticles_CUDA(float3 *pos, float3 *vel, int num, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	pos[i] += para->sph_dt*vel[i];
	return;
}

__global__ void enforceBoundary_CUDA(float3 *pos, float3 *vel, int num, float3 spaceSize)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	if(pos[i].x <= spaceSize.x*.001f)		{pos[i].x = spaceSize.x*.001f;	vel[i].x = fmaxf(vel[i].x, 0.0f);}
	if(pos[i].x >= spaceSize.x*.999f)	{pos[i].x = spaceSize.x*.999f;	vel[i].x = fminf(vel[i].x, 0.0f);}
	if(pos[i].y <= spaceSize.y*.001f)		{pos[i].y = spaceSize.y*.001f;	vel[i].y = fmaxf(vel[i].y, 0.0f);}
	if(pos[i].y >= spaceSize.y*.999f)	{pos[i].y = spaceSize.y*.999f;	vel[i].y = fminf(vel[i].y, 0.0f);}
	if(pos[i].z <= spaceSize.z*.001f)		{pos[i].z = spaceSize.z*.001f;	vel[i].z = fmaxf(vel[i].z, 0.0f);}
	if(pos[i].z >= spaceSize.z*.999f)	{pos[i].z = spaceSize.z*.999f;	vel[i].z = fminf(vel[i].z, 0.0f);}
	return;
}

struct compare_float3 {
	__host__ __device__
		float3 operator()(const float3& x, const float3& y) const {
		if (dot(x, x)>dot(y, y)) {
			return x;
		}
		else
			return y;
	}
};

void SPH::moveParticles()
{
// 	moveParticles_CUDA<<<gridsize, BLOCK_SIZE>>>(pos_fluid, vel_fluid, num_fluid, para);
	thrust::transform(thrust::device, vel_fluid, vel_fluid+num_fluid, pos_fluid, pos_fluid, saxpy_functor<float, float3>(para_h->sph_dt));

	int gridsize = (num_fluid-1)/BLOCK_SIZE+1;
	enforceBoundary_CUDA<<<gridsize, BLOCK_SIZE>>>(pos_fluid, vel_fluid, num_fluid, spaceSize);

	// 测量最大速度
	static float vmax = 0.0f;
	float3 t = thrust::reduce(thrust::device, vel_fluid, vel_fluid+num_fluid, make_float3(0.0f), compare_float3());
	vmax = fmaxf(vmax, length(t));
	printf("++++++++++max_vel=%f+++++++++++\n", vmax);

	return;
}

#ifdef DFSPH

__device__ void contributeDensityAlpha(float *density, float3 *term1, float *term2, float3 *pos_i, float3 *pos, float *mass, int j, int cellEnd, bool isBoundary, SPH_ParameterSet *para)
{
	while(j<cellEnd)
	{
		*density += mass[j] * cubic_spline_kernel(length(*pos_i - pos[j]), para);
		*term1 += mass[j]*cubic_spline_kernel_gradient(*pos_i-pos[j], para);
		if(!isBoundary)
			*term2 += dot(mass[j]*cubic_spline_kernel_gradient(*pos_i-pos[j], para), mass[j]*cubic_spline_kernel_gradient(*pos_i-pos[j], para));
		j++;
	}
	return;
}

__global__ void computeDensityAlpha_CUDA(float *density, float *alpha, float3 *pos_fluid, float *mass_fluid, int num, int *cellStart_fluid, int3 cellSize,
								float3* pos_staticBoundary, float *mass_staticBoundary, int *cellStart_staticBoundary, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	__shared__ float3 term1[BLOCK_SIZE];
	__shared__ float term2[BLOCK_SIZE];
	__shared__ float den[BLOCK_SIZE];
	term1[threadIdx.x] = make_float3(0.0f);
	term2[threadIdx.x] =0.0f;
	den[threadIdx.x]=0.0f;
	int cellID;
	__syncthreads();
#pragma unroll
	for(int m=0; m<27;	__syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_fluid[i]/para->sph_cell_length)+make_int3(m/9-1,(m%9)/3-1,m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributeDensityAlpha(&den[threadIdx.x], &term1[threadIdx.x], &term2[threadIdx.x], &pos_fluid[i], pos_fluid, mass_fluid, cellStart_fluid[cellID], cellStart_fluid[cellID+1], false, para);
		contributeDensityAlpha(&den[threadIdx.x], &term1[threadIdx.x], &term2[threadIdx.x], &pos_fluid[i], pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary[cellID], cellStart_staticBoundary[cellID+1], true, para);
	}

	density[i] = den[threadIdx.x];
	alpha[i] = -1.0f/fmaxf(para->sph_epsilon,(dot(term1[threadIdx.x],term1[threadIdx.x])+term2[threadIdx.x]));
	return ;
}

void SPH::computeDensityAlpha()
{
	int gridsize = (num_fluid - 1) / BLOCK_SIZE + 1;
	CUDA_CALL(cudaMemset(density, 0, sizeof(float)*num_fluid));
	computeDensityAlpha_CUDA<<<gridsize, BLOCK_SIZE>>>(density, alpha, pos_fluid, mass_fluid, num_fluid, cellStart_fluid, cellSize, pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary, para);


//     static float *density_h  = (float*)malloc(sizeof(float)*num_fluid);
//     cudaMemcpy(density_h, density, sizeof(float)*num_fluid, cudaMemcpyDeviceToHost);
//     float *max_den = thrust::max_element(density_h, density_h+num_fluid);
// 	float *min_den = thrust::min_element(density_h, density_h+num_fluid);
// 	float mean_den = thrust::reduce(density_h, density_h+num_fluid)/num_fluid;
//     printf("max density = %.12f\t min density = %.12f\t avg density = %.12f\n", *max_den, *min_den, mean_den); 
	return;
}

__device__ void contributeCohesion(float3 *a, int i, int cellID, float3 *pos, float *density, float *mass, float3 *color_grad, int j, int cellEnd, SPH_ParameterSet *para)
{
	while(j<cellEnd)
	{
		// surface tension
		*a += 2.0f*para->sph_rho0/(density[i]+density[j])*para->sph_color_energy_coefficient*mass[j]* surface_tension_kernel_gradient(pos[i]-pos[j], para);
		// surface area minimization
		*a += 2.0f*para->sph_rho0/(density[i]+density[j])*0.0001f*para->sph_color_energy_coefficient*(color_grad[i]-color_grad[j]);
		j++;
	}
	return;
}

__global__ void cohesion_CUDA(float3 *vel, float3 *pos_fluid, float *density, float *mass_fluid, float3 *color_grad, int num, int *cellStart, int3 cellSize, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	float3 a=make_float3(0.0f);
	int cellID;
#pragma unroll
	for(int m=0; m<27; __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_fluid[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributeCohesion(&a, i, cellID, pos_fluid, density, mass_fluid, color_grad, cellStart[cellID], cellStart[cellID+1], para);
	}
	vel[i] += a*para->sph_dt;
	return ;
}

void SPH::applyNonPressureForce()
{
	int gridsize = (num_fluid-1)/BLOCK_SIZE+1;

// 	// 采用XSPH来仿真黏性 [Ghost SPH for Animating Water]
// 	xsphViscosity_CUDA <<<gridsize, BLOCK_SIZE>>>(vel_fluid, pos_fluid, mass_fluid, density, num_fluid, cellStart_fluid, cellSize, para);

	viscosity_CUDA<<<gridsize, BLOCK_SIZE>>>(buffer_float3, vel_fluid, pos_fluid, mass_fluid, density, num_fluid, cellStart_fluid, cellSize, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();
	thrust::transform(thrust::device, vel_fluid, vel_fluid+num_fluid, buffer_float3, vel_fluid, thrust::plus<float3>());
	
  	//externalForce_CUDA <<<gridsize, BLOCK_SIZE>>>(vel_fluid, num_fluid, para);
	thrust::transform(thrust::device, vel_fluid, vel_fluid+num_fluid, vel_fluid, saxpb_functor<float,float3>(1.0f, para_h->sph_dt*para_h->sph_g));

	// 边界的处理方法参考[siggraph12][Versatile Rigid-Fluid Coupling for Incompressible SPH]
	// 其实如果不是双向耦合的话，从边界到流体的摩擦力并不能很明显的观察到
	//boundaryFriction_CUDA<<<gridsize, BLOCK_SIZE>>>(fluidParticles, density, fluidParticleNum, cellStart_fluid, cellSize, staticBoundaryParticles, cellStart_staticBoundary);

	// 表面张力和空气压力的处理方法参考[tog14][Robust Simulation of Small-Scale Thin Features in SPH-based Free Surface Flows]
	surfaceTensionAndAirPressure_CUDA<<<gridsize, BLOCK_SIZE>>>(vel_fluid, pos_fluid, density, mass_fluid, color_grad, num_fluid, cellStart_fluid, cellSize, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();
	//cohesion_CUDA<<<gridsize, BLOCK_SIZE>>>(vel_fluid, pos_fluid, density, mass_fluid, color_grad, num_fluid, cellStart_fluid, cellSize, para);
	return;
}

__device__ void contributeDivergenceError_fluid(float *e, int i, float3 *pos, float3 *vel, float *mass, int j, int cellEnd, SPH_ParameterSet *para)
{
	while(j<cellEnd)
	{
		*e += mass[j]*dot((vel[i]-vel[j]), cubic_spline_kernel_gradient(pos[i]-pos[j], para));
		j++;
	}
	return;
}

__device__ void contributeDivergenceError_boundary(float *e, float3 *pos_i, float3 *vel_i, float3 *pos, float *mass, int j, int cellEnd, SPH_ParameterSet *para)
{
	while (j < cellEnd)
	{
		*e += mass[j] * dot((*vel_i), cubic_spline_kernel_gradient(*pos_i - pos[j], para));
		j++;
	}
	return;
}

__global__ void computeDivergenceError_CUDA(float *error, float *stiff, float3 *pos_fluid, float3 *vel_fluid, float *mass_fluid, float *density, int num, float *alpha, int *cellStart_fluid, int3 cellSize,
										float3 *pos_staticBoundary, float *mass_staticBoundary, int *cellStart_staticBoundary, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	int cellID;
	__shared__ float e[BLOCK_SIZE];
	e[threadIdx.x] = 0.0f;
	__syncthreads();
#pragma unroll
	for(int m=0; m<27; __syncthreads(),m++)
	{
		cellID = particlePos2cellIdx( make_int3(pos_fluid[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributeDivergenceError_fluid(&e[threadIdx.x], i, pos_fluid, vel_fluid, mass_fluid, cellStart_fluid[cellID], cellStart_fluid[cellID+1], para);
		contributeDivergenceError_boundary(&e[threadIdx.x], &pos_fluid[i], &vel_fluid[i], pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary[cellID], cellStart_staticBoundary[cellID+1], para);	
	}
	//error[i] = fmaxf(0.0f,e[threadIdx.x]);
	// clamp:如果密度小于静止密度，就可以压缩
	if(density[i]+para->sph_dt*error[i]<para->sph_rho0)
		error[i] = (density[i]<=para->sph_rho0) ? 0.0f : error[i] = (para->sph_rho0-density[i])/para->sph_dt;	//若dt前后密度都小于静止密度，则取消密度限制；若dt前密度大于静止密度，则把error改为密度变化率
	else 
		error[i] = fmaxf(0.0f, error[i]);
	__syncthreads();
	stiff[i] = error[i]*alpha[i];
	return ;
}

__device__ void contributeAcceleration_fluid(float3 *a, int i, float3 *pos, float *mass, float *stiff, int j, int cellEnd, SPH_ParameterSet *para)
{
	while(j<cellEnd)
	{
		*a += mass[j]*(stiff[i]+stiff[j])* cubic_spline_kernel_gradient(pos[i]-pos[j], para);
		j++;
	}
	return;
}

__device__ void contributeAcceleration_boundary(float3 *a, float3 *pos_i, float3 *pos, float *mass, float stiff_i, int j, int cellEnd, SPH_ParameterSet *para)
{
	while(j<cellEnd)
	{ 
		*a += mass[j]*(stiff_i)* cubic_spline_kernel_gradient(*pos_i-pos[j], para);
		j++;
	}
	return;
}

__global__ void correctDivergenceError_CUDA(float3 *vel_fluid, float3 *pos_fluid, float *mass_fluid, float *stiff, int num, int *cellStart_fluid, int3 cellSize,
											float3 *pos_staticBoundary, float *mass_staticBoundary, int *cellStart_staticBoundary, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	float3 a = make_float3(0.0f);
	int cellID;
	__syncthreads();
#pragma unroll
	for(int m=0; m<27; __syncthreads(),m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_fluid[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributeAcceleration_fluid(&a, i, pos_fluid, mass_fluid, stiff, cellStart_fluid[cellID], cellStart_fluid[cellID+1], para);
		contributeAcceleration_boundary(&a, &pos_fluid[i], pos_staticBoundary, mass_staticBoundary, stiff[i], cellStart_staticBoundary[cellID], cellStart_staticBoundary[cellID+1], para);
		
	}
	
	vel_fluid[i] += a; // δt已经乘在a里面了
	return ;
}

template <typename T>
struct abs_plus
{
	__host__ __device__ T operator()(const T &lhs, const T &rhs) const {return abs(lhs) + abs(rhs);}
};

void SPH::correctDivergenceError()
{
	int gridsize = (num_fluid-1)/BLOCK_SIZE+1;
	float avgError = para_h->sph_divergence_avg_threshold*para_h->sph_rho0+1.0f;
	int iter = 0;

	computeDivergenceError_CUDA <<<gridsize, BLOCK_SIZE >>>(error, stiff_div, pos_fluid, vel_fluid, mass_fluid, density, num_fluid, alpha, cellStart_fluid, cellSize, pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary, para);
	while ((iter<1 || avgError>para_h->sph_divergence_avg_threshold*para_h->sph_rho0) && iter<para_h->sph_max_iter)
	{
		correctDivergenceError_CUDA<<<gridsize, BLOCK_SIZE>>>(vel_fluid, pos_fluid, mass_fluid, stiff_div, num_fluid, cellStart_fluid, cellSize, pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary, para);
		computeDivergenceError_CUDA <<<gridsize, BLOCK_SIZE >>>(error, stiff_div, pos_fluid, vel_fluid, mass_fluid, density, num_fluid, alpha, cellStart_fluid, cellSize, pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary, para);
		iter++;
		avgError = thrust::reduce(thrust::device, error, error+num_fluid, 0.0f, abs_plus<float>())/num_fluid;
	}
	sprintf(frame_log.str[frame_log.ptr++], "correct divergence %d iterations\tavgError = %e", iter, avgError);
	return;
}

__device__ void contributeDensityError_fluid(float *e, int i, float3 *pos, float3 *vel, float *mass, int j, int cellEnd, SPH_ParameterSet *para)
{
	while(j<cellEnd)
	{ 
		*e += mass[j]*dot((vel[i]-vel[j]), cubic_spline_kernel_gradient(pos[i]-pos[j], para));
		j++;
	}
	return;
}

__device__ void contributeDensityError_boundary(float *e, float3 *vel_i, float3 *pos_i, float3 *pos, float *mass, int j, int cellEnd, SPH_ParameterSet *para)
{
	while (j < cellEnd)
	{
		*e += mass[j] * dot((*vel_i), cubic_spline_kernel_gradient(*pos_i-pos[j], para));
		j++;
	}
	return;
}

__global__ void computeDensityError_CUDA(float *error, float *stiff, float3 *pos_fluid, float3 *vel_fluid, float *mass_fluid, int num, float *density, float *alpha, int *cellStart_fluid, int3 cellSize,
											float3 *pos_staticBoundary, float *mass_staticBoundary, int *cellStart_staticBoundary, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	__shared__ float e[BLOCK_SIZE];
	e[threadIdx.x]=0.0f;
	int cellID;
	__syncthreads();
#pragma unroll
	for(int m=0; m<27; __syncthreads(),m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_fluid[i]/para->sph_cell_length)+make_int3(m/9-1,(m%9)/3-1,m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributeDensityError_fluid(&e[threadIdx.x], i, pos_fluid, vel_fluid, mass_fluid, cellStart_fluid[cellID], cellStart_fluid[cellID+1], para);
		contributeDensityError_boundary(&e[threadIdx.x], &vel_fluid[i], &pos_fluid[i], pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary[cellID], cellStart_staticBoundary[cellID+1], para);
	}

	//clamp
	error[i] = fmaxf(0.0f, para->sph_dt*e[threadIdx.x] + density[i] - para->sph_rho0);
	stiff[i] = error[i]*alpha[i];
	return ;
}

__global__ void correctDensityError_CUDA(float3 *vel_fluid, float3 *pos_fluid, float *mass_fluid, float *stiff, int num, int *cellStart_fluid, int3 cellSize,
										float3 *pos_staticBoundary, float *mass_staticBoundary, int *cellStart_staticBoundary, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	__shared__ float3 a[BLOCK_SIZE];
	a[threadIdx.x]=make_float3(0.0f);
	int cellID;
	__syncthreads();
#pragma unroll
	for(int m=0; m<27; __syncthreads(),m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_fluid[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributeAcceleration_fluid(&a[threadIdx.x], i, pos_fluid, mass_fluid, stiff, cellStart_fluid[cellID], cellStart_fluid[cellID+1], para);
		contributeAcceleration_boundary(&a[threadIdx.x], &pos_fluid[i], pos_staticBoundary, mass_staticBoundary, stiff[i], cellStart_staticBoundary[cellID], cellStart_staticBoundary[cellID+1], para);
	}

	vel_fluid[i] += a[threadIdx.x]/para->sph_dt;
	return ;
}

void SPH::correctDensityError()
{
	int gridsize = (num_fluid-1)/BLOCK_SIZE+1;
	float avgError = para_h->sph_density_avg_threshold*para_h->sph_rho0+1.0f;
	int iter = 0;

	computeDensityError_CUDA <<<gridsize, BLOCK_SIZE >>>(error, stiff_den, pos_fluid, vel_fluid, mass_fluid, num_fluid, density, alpha, cellStart_fluid, cellSize, pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary, para);

	while((iter<2 || avgError>para_h->sph_density_avg_threshold*para_h->sph_rho0) && iter<para_h->sph_max_iter)
	{
		correctDensityError_CUDA<<<gridsize, BLOCK_SIZE>>>(vel_fluid, pos_fluid, mass_fluid, stiff_den, num_fluid, cellStart_fluid, cellSize, pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary, para);
		computeDensityError_CUDA <<<gridsize, BLOCK_SIZE >>>(error, stiff_den, pos_fluid, vel_fluid, mass_fluid, num_fluid, density, alpha, cellStart_fluid, cellSize, pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary, para);
		iter++;
		if(iter>=2)
			avgError = thrust::reduce(thrust::device, error, error+num_fluid, 0.0f, abs_plus<float>())/ num_fluid;
	}
	sprintf(frame_log.str[frame_log.ptr++], "correct density %d iterations\tavgError = %e", iter, avgError);
	return; 
}
#endif

__device__ void contributePos_fluid(float3 *p, float *d, int i, float3 *pos, int j, int cellEnd, SPH_ParameterSet *para)
{
	while(j<cellEnd)
	{
		*d += cubic_spline_kernel(length(pos[i]-pos[j]), para);
		*p += pos[j]*cubic_spline_kernel(length(pos[i]-pos[j]), para);
		j++;
	}
	return;
}

__device__ void contributePos_boundary(float3 *p, float *d, float3 *pos_i, float3 *pos, int j, int cellEnd, SPH_ParameterSet *para)
{
	while(j < cellEnd)
	{
		*d += cubic_spline_kernel(length(*pos_i-pos[j]), para);
		*p += pos[j]*cubic_spline_kernel(length(*pos_i-pos[j]), para);
		j++;
	}
	return;
}

__global__ void computeMeanPos_CUDA(float3 *mean_pos, float3 *pos_fluid, int num, int *cellStart_fluid, int3 cellSize,
	float3 *pos_staticBoundary, int *cellStart_staticBoundary, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	__shared__ float3 p[BLOCK_SIZE];
	__shared__ float d[BLOCK_SIZE];
	p[threadIdx.x]=make_float3(0.0f);
	d[threadIdx.x]=0.0f;
	int cellID;
	__syncthreads();
#pragma unroll
	for(int m=0; m<27; __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_fluid[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributePos_fluid(&p[threadIdx.x], &d[threadIdx.x], i, pos_fluid, cellStart_fluid[cellID], cellStart_fluid[cellID+1], para);
		contributePos_boundary(&p[threadIdx.x], &d[threadIdx.x], &pos_fluid[i], pos_staticBoundary, cellStart_staticBoundary[cellID], cellStart_staticBoundary[cellID+1], para);
	}
	if(d[threadIdx.x]<para->sph_epsilon)
		mean_pos[i] = pos_fluid[i];
	else
		mean_pos[i] = p[threadIdx.x]/d[threadIdx.x];
	return ;
}

__device__ void contributeCovariance(square3 *cov, float *d, float3 *mean_pos_i, float3 *pos, int j, int cellEnd, SPH_ParameterSet *para)
{
	float3 v;
	while(j<cellEnd)
	{
		v = *mean_pos_i-pos[j];
		(*cov).x += make_float3(v.x*v.x, v.y*v.x, v.z*v.x) * cubic_spline_kernel(length(v), para);
		(*cov).y += make_float3(v.x*v.y, v.y*v.y, v.z*v.y) * cubic_spline_kernel(length(v), para);
		(*cov).z += make_float3(v.x*v.z, v.y*v.z, v.z*v.z) * cubic_spline_kernel(length(v), para);
		*d += cubic_spline_kernel(length(v), para);
		j++;
	}
	return;
}

__global__ void computeCovariance_CUDA(square3 *covariance, float3 *mean_pos, float3 *pos_fluid, int num, int *cellStart_fluid, int3 cellSize,
	float3 *pos_staticBoundary, int *cellStart_staticBoundary, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	float d=0.0f;
	__shared__ square3 cov[BLOCK_SIZE];
	cov[threadIdx.x].x = make_float3(0.0f);
	cov[threadIdx.x].y = make_float3(0.0f);
	cov[threadIdx.x].z = make_float3(0.0f);
	int cellID;
	__syncthreads();
#pragma unroll
	for(int m=0; m<27; __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_fluid[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributeCovariance(&cov[threadIdx.x], &d, &mean_pos[i], pos_fluid, cellStart_fluid[cellID], cellStart_fluid[cellID+1], para);
		contributeCovariance(&cov[threadIdx.x], &d, &mean_pos[i], pos_staticBoundary, cellStart_staticBoundary[cellID], cellStart_staticBoundary[cellID+1], para);
	}
	covariance[i].x = cov[threadIdx.x].x/fmaxf(para->sph_epsilon,d);
	covariance[i].y = cov[threadIdx.x].y/fmaxf(para->sph_epsilon,d);
	covariance[i].z = cov[threadIdx.x].z/fmaxf(para->sph_epsilon,d);
	return ;
}

inline __device__ void transpose3(square3 *dst, square3 *src)
{
	(*dst).x = make_float3(src->x.x, src->y.x, src->z.x);
	(*dst).y = make_float3(src->x.y, src->y.y, src->z.y);
	(*dst).z = make_float3(src->x.z, src->y.z, src->z.z);
	return;
}

inline __device__ void qr3(square3 *dst, square3 *src)
{
	square3 tmp;
	tmp.x=src->x;
	tmp.y=src->y;
	tmp.z=src->z;
	//orthonormalize
	src->x /= length(src->x);
	src->y -= dot(src->y, src->x)*src->x;
	src->y /= length(src->y);
	src->z -= dot(src->z, src->x)*src->x;
	src->z -= dot(src->z, src->y)*src->y;
	src->z /= length(src->z);
	//qr
	dst->x = make_float3(dot(tmp.x,src->x), 0.0f, 0.0f);
	dst->y = make_float3(dot(tmp.y,src->x), dot(tmp.y, src->y), 0.0f);
	dst->z = make_float3(dot(tmp.z,src->x), dot(tmp.z, src->y), dot(tmp.z, src->z));
	return;
}

__global__ void computeSingularValue_CUDA(float3 *sv, square3 *mat, int num, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	square3 s1,s2;
	float tmp;
	transpose3(&s1, &mat[i]);
#pragma unroll
	for(int j=0; j<para->sph_svd_iter; j++)
	{
		transpose3(&s2, &s1);
		qr3(&s1, &s2);
	}
	sv[i] = make_float3(s1.x.x, s1.y.y, s1.z.z);
	
	//sort
	if(sv[i].x < sv[i].y)
	{
		tmp = sv[i].x;
		sv[i].x = sv[i].y;
		sv[i].y = tmp;
	}
	if(sv[i].y < sv[i].z)
	{
		if(sv[i].x < sv[i].z)
		{
			tmp = sv[i].z;
			sv[i].z = sv[i].y;
			sv[i].y = sv[i].x;
			sv[i].x = tmp;
		}
		else
		{
			tmp = sv[i].y;
			sv[i].y = sv[i].z;
			sv[i].z = tmp;
		}
	}
	return;
}

// __global__ void computePossibleBoundary_CUDA(float *possibleBoundary, float *density, float3 *ev, int num)
// {
// 	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
// 	if(i>=num) return;
// 	possibleBoundary[i] = 1.0f - ev[i].z/fmaxf(sph_epsilon, ev[i].x);
// 	possibleBoundary[i] = int(possibleBoundary[i]+0.6f);
// 	return;
// }

__device__ void contributeColorGrad_fluid(float3 *numerator, float *denominator, int i, float3 *pos, float *mass, float *density, int j, int cellEnd, SPH_ParameterSet *para)
{
	while(j<cellEnd)
	{ 
		*numerator += mass[j]/para->sph_rho0 * cubic_spline_kernel_gradient(pos[i]-pos[j], para);
		*denominator += mass[j]/para->sph_rho0*cubic_spline_kernel(length(pos[i]-pos[j]), para);
		j++;
	}
	return;
}

__device__ void contributeColorGrad_boundary(float3 *numerator, float *denominator, float3 *pos_i, float3 *pos, float *mass, int j, int cellEnd, SPH_ParameterSet *para)
{
	while(j<cellEnd)
	{
		*numerator += mass[j]/para->sph_rhob * cubic_spline_kernel_gradient(*pos_i-pos[j], para);
		*denominator += mass[j]/para->sph_rhob*cubic_spline_kernel(length(*pos_i-pos[j]), para);
		j++;
	}
	return;
}

__global__ void computeColorGrad_CUDA(float3* color_grad, float3 *pos_fluid, float *mass_fluid, float *density, int num, int *cellStart_fluid, int3 cellSize,
	float3 *pos_staticBoundary, float *mass_staticBoudnary, int *cellStart_staticBoundary, SPH_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i>=num) return;
	float3 c_g = make_float3(0.0f);
	float denominator = 0.0f; // 分母
	int cellID;
#pragma unroll
	for(int m=0; m<27; __syncthreads(),m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_fluid[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		contributeColorGrad_fluid(&c_g, &denominator, i, pos_fluid, mass_fluid, density, cellStart_fluid[cellID], cellStart_fluid[cellID+1], para);
		contributeColorGrad_boundary(&c_g, &denominator, &pos_fluid[i], pos_staticBoundary, mass_staticBoudnary, cellStart_staticBoundary[cellID], cellStart_staticBoundary[cellID+1], para);
	}

	color_grad[i] = c_g /fmaxf(para->sph_epsilon, denominator);
	return ;
}

__global__ void computeProbableSurface_CUDA(float *probableSurface, float3 *color_grad, int num)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	probableSurface[i] = 0.001f *dot( color_grad[i], color_grad[i]);
	return;
}

void SPH::surfaceDetection()
{
	int gridsize = (num_fluid-1)/BLOCK_SIZE+1;
// 	computeMeanPos_CUDA <<<gridsize, BLOCK_SIZE >>>(mean_pos, pos_fluid, num_fluid, cellStart_fluid, cellSize, pos_staticBoundary, cellStart_staticBoundary);
// 	cudaDeviceSynchronize(); CHECK_KERNEL();
// 	computeCovariance_CUDA <<<gridsize, BLOCK_SIZE >>>(covariance, mean_pos, pos_fluid, num_fluid, cellStart_fluid, cellSize, pos_staticBoundary, cellStart_staticBoundary);
// 	cudaDeviceSynchronize(); CHECK_KERNEL();
// 	computeSingularValue_CUDA<<<gridsize, BLOCK_SIZE >>>(singular, covariance, num_fluid);
// 	cudaDeviceSynchronize(); CHECK_KERNEL();
// 	computePossibleBoundary_CUDA<<<gridsize, BLOCK_SIZE >>>(possibleBoundary, density, singular, num_fluid);
// 	cudaDeviceSynchronize(); CHECK_KERNEL();

	computeColorGrad_CUDA<<<gridsize, BLOCK_SIZE >>>(color_grad, pos_fluid, mass_fluid, density, num_fluid, cellStart_fluid, cellSize, pos_staticBoundary, mass_staticBoundary, cellStart_staticBoundary, para);
	computeProbableSurface_CUDA<<<gridsize, BLOCK_SIZE >>>(probableSurface, color_grad, num_fluid);
	return;
}

__global__ void calculateEnergy_CUDA(float *energy, float3 *v, float3 *fluid, float *mass, int num_fluid) {
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; // _i是fluid的下标
	if (i>=num_fluid) return;
	energy[i] = dot(v[i], v[i])*mass[i]*0.5f + mass[i]*fluid[i].y*9.8f;
	return;
}

void SPH::calculateEnergy() {
	calculateEnergy_CUDA<<<(num_fluid-1)/BLOCK_SIZE+1, BLOCK_SIZE>>>(buffer_float, vel_fluid, pos_fluid, mass_fluid, num_fluid);
	cudaDeviceSynchronize();
	float r = thrust::reduce(thrust::device, buffer_float, buffer_float+num_fluid);
	printf("+++++++++++++++++++total energy = %f++++++++++++++++++++\n", r);
	FILE *fp = fopen("energy.txt", "a");
	fprintf(fp, "%f\n", r);
	fclose(fp);
}

void SPH::calculateAverageDensity() {
	float r = thrust::reduce(thrust::device, density, density+num_fluid);
	r /= num_fluid;
	printf("+++++++++++++++++++average density = %f++++++++++++++++++++\n", r);
	FILE *fp = fopen("density.txt", "a");
	fprintf(fp, "%f\n", r);
	fclose(fp);
}