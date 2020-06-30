#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include "global.h"
#include "SPH.h"
#include "FLIP.h"
#include "Voronoi3D.h"

__global__ void generate_dots_CUDA(float3* dot, float3* posColor, float3 *pos, float *density, int num)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (i >= num) return;

	dot[i] = pos[i]; 
	float color = density[i];
	if (color < 1.0f)
	{
		color = 1.0f - (1.0f - color)*1.0f;
		posColor[i] = make_float3(1.0f, color, color);
	}
	else
	{
		color = 1.0f - (color - 1.0f)*2.0f;
		posColor[i] = make_float3(color, color, 1.0f);
	}
	//posColor[i] = make_float3(1.0f, 1.0f-density[i], 1.0f-density[i]);
}

extern "C" 	void generate_dots(float3* dot, float3* posColor, SPH *sph)
{
	int gridsize = (sph->num_fluid - 1) / BLOCK_SIZE + 1;
	//generate_dots_CUDA <<<gridsize, BLOCK_SIZE >>>(dot, posColor, sph->pos_staticBoundary, sph->mass_staticBoundary, sph->num_staticBoundary);
	generate_dots_CUDA <<<gridsize, BLOCK_SIZE >>>(dot, posColor, sph->pos_fluid, sph->density, sph->num_fluid);
	cudaDeviceSynchronize();	CHECK_KERNEL();
	return;
}

__global__ void generate_dots_CUDA(float3* dot, float3* posColor, float3 *pos, float3 *vel, int num)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i >= num) return;

	dot[i] = pos[i];
	posColor[i] = make_float3(0.0f, 1.0f, 0.0f);
}

extern "C" 	void generate_dots_flip(float3* dot, float3* posColor, FLIP *flip)
{
	int gridsize = (flip->num_flip - 1) / BLOCK_SIZE + 1;
	generate_dots_CUDA <<<gridsize, BLOCK_SIZE >>>(dot, posColor, flip->pos_flip, flip->vel_flip, flip->num_flip);
	cudaDeviceSynchronize();	CHECK_KERNEL();
	return;
}

__global__ void generate_voro_CUDA(float3* dot, float3* posColor, int3 *edge, float3 *site, int *edge_flag, int num)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i >= num) return;

	dot[2*i] = site[edge[i].x];
	dot[2*i+1] = site[edge[i].y];
	float3 color;
	if(edge_flag[i]==1)
	{
		color = make_float3(1.0f, 0.0f, 0.0f);
	}
	else if(edge_flag[i]==2)
	{
		color = make_float3(0.0f, 1.0f, 0.0f);
	}
	else if(edge_flag[i]==3)
	{
		color = make_float3(0.0f, 0.0f, 1.0f);
	}
	else if(edge_flag[i]==4)
	{
		color = make_float3(1.0f, 0.0f, 1.0f);
	}
	posColor[2*i] = color;
	posColor[2*i+1] = color;
}

extern "C" 	void generate_voro(float3* dot, float3* posColor, Voronoi3D *voro)
{
	int gridsize = (voro->num_edge_h - 1) / BLOCK_SIZE + 1;
	generate_voro_CUDA <<<gridsize, BLOCK_SIZE >>>(dot, posColor, voro->edge, voro->site, voro->edge_flag, voro->num_edge_h);
	cudaDeviceSynchronize();	CHECK_KERNEL();
	return;
}