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
#include "FLIP.h"

__global__ void init_flip_CUDA(float3 *pos_flip, float3 *vel_flip, float3 *pos_sph, int num, FLIP_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	int sph_idx = i/8;

	pos_flip[i] = pos_sph[sph_idx];
	vel_flip[i] = make_float3(0.0f);
	__syncthreads();

	pos_flip[sph_idx*8+0] += make_float3(0.25f, 0.25f, 0.25f)*para->sph_spacing;
	pos_flip[sph_idx*8+1] += make_float3(0.25f, 0.25f, -0.25f)*para->sph_spacing;
	pos_flip[sph_idx*8+2] += make_float3(0.25f, -0.25f, 0.25f)*para->sph_spacing;
	pos_flip[sph_idx*8+3] += make_float3(0.25f, -0.25f, -0.25f)*para->sph_spacing;
	pos_flip[sph_idx*8+4] += make_float3(-0.25f, 0.25f, 0.25f)*para->sph_spacing;
	pos_flip[sph_idx*8+5] += make_float3(-0.25f, 0.25f, -0.25f)*para->sph_spacing;
	pos_flip[sph_idx*8+6] += make_float3(-0.25f, -0.25f, 0.25f)*para->sph_spacing;
	pos_flip[sph_idx*8+7] += make_float3(-0.25f, -0.25f, -0.25f)*para->sph_spacing;
	return;
}

void FLIP::init_flip(SPH *sph)
{
	CUDA_CALL(cudaMemset(vel_fluid_last, 0, sizeof(float3)*sph->num_fluid));
	int gridsize = (num_flip-1)/BLOCK_SIZE+1;
	init_flip_CUDA<<<gridsize, BLOCK_SIZE>>>(pos_flip, vel_flip, sph->pos_fluid, num_flip, para);
	neighborSearch();
	return;
}

inline __device__ int particlePos2cellIdx(int3 pos, int3 cellSize)
{
	// return (cellSize.x*cellSize.y*cellSize.z) if the particle is out of the grid
	return (pos.x>=0 && pos.x<cellSize.x && pos.y>=0 && pos.y<cellSize.y && pos.z>=0 && pos.z<cellSize.z)?
		(((pos.x*cellSize.y)+pos.y)*cellSize.z+pos.z)
		: (cellSize.x*cellSize.y*cellSize.z);
}

__global__ void mapParticles2cells_CUDA(int *particles2cells, int3 cellSize, float3 *pos, int num, FLIP_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	particles2cells[i] = particlePos2cellIdx(make_int3(pos[i]/para->sph_smoothing_radius), cellSize);
	return ;
}

extern "C" __global__ void countingInCell_CUDA(int* cellStart, int *particles2cells, int num);

void FLIP::neighborSearch()
{
	int gridsize = (num_flip-1)/BLOCK_SIZE+1;
	mapParticles2cells_CUDA<<<gridsize, BLOCK_SIZE>>>(particles2cells, cellSize, pos_flip, num_flip, para);
	cudaMemcpy(particles2cells2, particles2cells, sizeof(int)*num_flip, cudaMemcpyDeviceToDevice);
	thrust::sort_by_key(thrust::device, particles2cells, particles2cells + num_flip, pos_flip);
	cudaMemcpy(particles2cells, particles2cells2, sizeof(int)*num_flip, cudaMemcpyDeviceToDevice);
	thrust::sort_by_key(thrust::device, particles2cells, particles2cells + num_flip, vel_flip);

	cudaMemset(cellStart, 0, sizeof(int)*(cellSize.x*cellSize.y*cellSize.z+1));
	countingInCell_CUDA<<<gridsize, BLOCK_SIZE>>>(cellStart, particles2cells, num_flip);
	thrust::exclusive_scan(thrust::device, cellStart, cellStart+cellSize.x*cellSize.y*cellSize.z+1, cellStart);
	return;
}

inline __device__ float cubic_spline_kernel(float r, FLIP_ParameterSet *para)
{
	float q = 2.0f*fabs(r)/para->sph_smoothing_radius;
	return (q<para->flip_epsilon)?0.0f:
		((q)<=1.0f?(pow(2.0f-q, 3)-4.0f*pow(1.0f-q, 3)):
		(q)<=2.0f?(pow(2.0f-q, 3)):
		0.0f) / (4.0f*M_PI*pow(para->sph_smoothing_radius, 3));
}

inline __device__ float3 cubic_spline_kernel_gradient(float3 r, FLIP_ParameterSet *para)
{
	float q = 2.0f*length(r)/para->sph_smoothing_radius;
	return
		((q) <= 1.0f ? -(3.0f*(2.0f-q)*(2.0f-q)-12.0f*(1.0f-q)*(1.0f-q)) :
		(q) <= 2.0f ? -(3.0f*(2.0f-q)*(2.0f-q)) :
		0.0f) / (2.0f*M_PI*pow(para->sph_smoothing_radius, 4)) * r / fmaxf(para->flip_epsilon, length(r));
}

__global__ void flip2sphMapping_CUDA(float3 *vel_fluid, float3 *pos_fluid, float3 *vel_flip, float3 *pos_flip, int num, int3 cellSize, int *cellStart, FLIP_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	float3 v=make_float3(0.0f);
	float denominator = 0.0f;
	int cellID;
	__syncthreads();
#pragma unroll
	for(int m=0; m<27; __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_fluid[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		int j=cellStart[cellID], cellEnd = cellStart[cellID+1];
		while(j<cellEnd)
		{
			v += vel_flip[j] * cubic_spline_kernel(length(pos_fluid[i]-pos_flip[j]), para);
			denominator += cubic_spline_kernel(length(pos_fluid[i]-pos_flip[j]), para);
			j++;
		}
	}
	vel_fluid[i] = v/fmaxf(para->flip_epsilon, denominator);
	return ;
}

void FLIP::flip2sphMapping()
{
	int gridsize = (sph->num_fluid-1)/BLOCK_SIZE+1;
	flip2sphMapping_CUDA<<<gridsize,BLOCK_SIZE>>>(sph->vel_fluid, sph->pos_fluid, vel_flip, pos_flip, sph->num_fluid, cellSize, cellStart, para);
	return;
}

__global__ void sph2flipMapping_CUDA(float3 *vel_flip, float3 *pos_flip, float3 *vel_fluid, float3 *vel_fluid_last, float3 *pos_fluid, int num, int3 cellSize, int *cellStart_sph, FLIP_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	float3 v_pic=make_float3(0.0f);
	float3 v_flip=make_float3(0.0f);
	float denominator = 0.0f;
	int cellID;
	__syncthreads();
#pragma unroll
	for(int m=0; m<27; __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_flip[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		int j=cellStart_sph[cellID], cellEnd = cellStart_sph[cellID+1];
		while(j<cellEnd)
		{
			float r = length(pos_flip[i]-pos_fluid[j]);
			v_pic += vel_fluid[j] * cubic_spline_kernel(r, para);
			v_flip += (vel_fluid[j]-vel_fluid_last[j]) * cubic_spline_kernel(r, para);
			denominator += cubic_spline_kernel(r, para);
			j++;
		}
	}
	v_pic /= fmaxf(para->flip_epsilon, denominator);
	v_flip = vel_flip[i]+v_flip/fmaxf(para->flip_epsilon, denominator);

	if(denominator<para->flip_epsilon)
		vel_flip[i] += para->sph_dt*para->flip_g;
 	else
		vel_flip[i] = v_pic*para->pic_alpha + v_flip*(1.0f-para->pic_alpha);
	return ;
}

void FLIP::sph2flipMapping()
{
	int gridsize = (num_flip-1)/BLOCK_SIZE+1;
	sph2flipMapping_CUDA<<<gridsize,BLOCK_SIZE>>>(vel_flip, pos_flip, sph->vel_fluid, vel_fluid_last, sph->pos_fluid, num_flip, cellSize, sph->cellStart_fluid, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();
	return;
}

__global__ void enforceFlipBoundary_CUDA(float3 *pos, float3 *vel, int num, float3 spaceSize)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	if(pos[i].x < spaceSize.x*.01f)	{ pos[i].x = spaceSize.x*.01f;	vel[i].x = fmaxf(vel[i].x, 0.0f); }
	if(pos[i].x > spaceSize.x*.99f)	{ pos[i].x = spaceSize.x*.99f;	vel[i].x = fminf(vel[i].x, 0.0f); }
	if(pos[i].y < spaceSize.y*.01f)	{ pos[i].y = spaceSize.y*.01f;	vel[i].y = fmaxf(vel[i].y, 0.0f); }
	if(pos[i].y > spaceSize.y*.99)	{ pos[i].y = spaceSize.y*.99f;	vel[i].y = fminf(vel[i].y, 0.0f); }
	if(pos[i].z < spaceSize.z*.01f)	{ pos[i].z = spaceSize.z*.01f;	vel[i].z = fmaxf(vel[i].z, 0.0f); }
	if(pos[i].z > spaceSize.z*.99f)	{ pos[i].z = spaceSize.z*.99f;	vel[i].z = fminf(vel[i].z, 0.0f); }
	return;
}

__global__ void positionCorrection_CUDA(float3 *pos_correction, float3 *pos_flip, int num, int3 cellSize, int *cellStart, FLIP_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	float3 dp=make_float3(0.0f);
	int cellID;
	__syncthreads();
#pragma unroll
	for(int m=0; m<27; __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos_flip[i]/para->sph_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		int j=cellStart[cellID], cellEnd = cellStart[cellID+1];
		while(j<cellEnd)
		{
			if(length(pos_flip[j]-pos_flip[i])>0.1f*para->sph_smoothing_radius)
			dp -= para->flip_mass * normalize(pos_flip[j]-pos_flip[i])*cubic_spline_kernel(length(pos_flip[j]-pos_flip[i]),para);
			//dp -= para->flip_mass * cubic_spline_kernel_gradient(pos_flip[i]-pos_flip[j], para);
			j++;
		}
	}
	pos_correction[i] = dp;
	return ;
}

void FLIP::flipAdvect()
{
	int gridsize = (num_flip-1)/BLOCK_SIZE+1;
	thrust::transform(thrust::device, vel_flip, vel_flip+num_flip, pos_flip, pos_flip, saxpy_functor<float, float3>(para_h->sph_dt));
// 	enforceFlipBoundary_CUDA<<<gridsize, BLOCK_SIZE>>>(pos_flip, vel_flip, num_flip, spaceSize);
// 	neighborSearch();

	// anisotropic position correction by [2012][TVCG][Preserving Fluid Sheets with Adaptively Sampled Anisotropic Particles]
	//positionCorrection_CUDA<<<gridsize, BLOCK_SIZE>>>(pos_correction, pos_flip, num_flip, cellSize, cellStart, para);
	//thrust::transform(thrust::device, pos_correction, pos_correction+num_flip, pos_flip, pos_flip, saxpy_functor<float,float3>(para_h->flip_position_correction_gamma*para_h->sph_dt*para_h->sph_smoothing_radius));
	enforceFlipBoundary_CUDA<<<gridsize, BLOCK_SIZE>>>(pos_flip, vel_flip, num_flip, spaceSize);
	neighborSearch();
	return;
}