#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include <thrust/pair.h>
#include <thrust/execution_policy.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "global.h"
#include "MC.h"
#include "tables.h"

extern struct log frame_log;
// textures containing look-up tables
texture<uint, 1, cudaReadModeElementType> edgeTex;
texture<uint, 1, cudaReadModeElementType> triTex;
texture<uint, 1, cudaReadModeElementType> numVertsTex;

inline __device__ int particlePos2cellIdx(int3 pos, int3 cellSize)
{
	// return (cellSize.x*cellSize.y*cellSize.z) if the particle is out of the grid
	return (pos.x>=0 && pos.x<cellSize.x && pos.y>=0 && pos.y<cellSize.y && pos.z>=0 && pos.z<cellSize.z)?
		(((pos.x*cellSize.y)+pos.y)*cellSize.z+pos.z)
		: (cellSize.x*cellSize.y*cellSize.z);
}

__global__ void mapParticles2cells_CUDA(int *particles2cells, int3 cellSize, float3 *pos, int num, MC_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	particles2cells[i] = particlePos2cellIdx(make_int3(pos[i]/para->mc_smoothing_radius), cellSize);
	return ;
}

extern "C" __global__ void countingInCell_CUDA(int* cellStart, int *particles2cells, int num);

void MC::mapParticles2Cells()
{
	int gridsize = (num_particle-1)/BLOCK_SIZE+1;
	mapParticles2cells_CUDA<<<gridsize, BLOCK_SIZE>>>(particles2cells, SMCellSize, particlePos, num_particle, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();

	thrust::sort_by_key(thrust::device, particles2cells, particles2cells + num_particle, particlePos);

	cudaMemset(cellStart, 0, sizeof(int)*(SMCellSize.x*SMCellSize.y*SMCellSize.z+1));
	countingInCell_CUDA<<<gridsize, BLOCK_SIZE>>>(cellStart, particles2cells, num_particle);
	cudaDeviceSynchronize(); CHECK_KERNEL();
	thrust::exclusive_scan(thrust::device, cellStart, cellStart+SMCellSize.x*SMCellSize.y*SMCellSize.z+1, cellStart);
	return;
}



inline __device__ float mc_kernel(float r2, MC_ParameterSet *para)
{
	return pow((1.0f- r2/(para->mc_smoothing_radius*para->mc_smoothing_radius) ),3);
}

inline __device__ float3 mc_kernel_gradient(float3 r, MC_ParameterSet *para)
{
	return 
		-6.0f*r* pow((dot(r,r)/(para->mc_smoothing_radius*para->mc_smoothing_radius) - 1), 2)/(para->mc_smoothing_radius*para->mc_smoothing_radius);
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
	dst->x = make_float3(dot(tmp.x, src->x), 0.0f, 0.0f);
	dst->y = make_float3(dot(tmp.y, src->x), dot(tmp.y, src->y), 0.0f);
	dst->z = make_float3(dot(tmp.z, src->x), dot(tmp.z, src->y), dot(tmp.z, src->z));
	return;
}

__device__ float implicitFunction(float3 x, int3 cellSize, int *cellStart, float3 *pos, MC_ParameterSet *para)
{
	float3 hat_x = make_float3(0.0f);
	float denominator = 0.0f;
// 	square3 gradient_hat_x;
// 	float3 u,v;
// 	float ev;
// 	gradient_hat_x.x = make_float3(0.0f);
// 	gradient_hat_x.y = make_float3(0.0f);
// 	gradient_hat_x.z = make_float3(0.0f);
 	//step1: 计算\hat{x} 和 \gradient_x \hat{x}
	int cellID;
	int j;
#pragma unroll
	for(int m=0; m<27; m++)
	{
		cellID = particlePos2cellIdx(make_int3(x/para->mc_smoothing_radius)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		for (j=cellStart[cellID]; j<cellStart[cellID+1]; j++)
		{
			if(mc_kernel(dot(x-pos[j],x-pos[j]), para)>0.0f)
			{
				denominator += mc_kernel(dot(x-pos[j],x-pos[j]), para);
				hat_x += pos[j] * mc_kernel(dot(x-pos[j],x-pos[j]), para);
			}
// 			//u 用做临时变量
// 			u = mc_kernel_gradient(x-pos[j], para);
// 			gradient_hat_x.x += pos[j] * u.x;
// 			gradient_hat_x.y += pos[j] * u.y;
// 			gradient_hat_x.z += pos[j] * u.z;
		}
	}
	if(denominator< para->mc_epsilon) return para->mc_smoothing_radius;
	hat_x /= denominator;
// 	gradient_hat_x.x /= fmaxf(para->mc_epsilon, denominator);
// 	gradient_hat_x.y /= fmaxf(para->mc_epsilon, denominator);
// 	gradient_hat_x.z /= fmaxf(para->mc_epsilon, denominator);
// 	// step2: 幂法计算最大EV
// 	u = make_float3(1.0f);
// 	denominator = 1.0f; //临时变量
// 	ev = 0.0f;
// 	for( k=0; k<para->mc_ev_iter && fabs(ev-denominator)>para->mc_epsilon; k++)
// 	{
// 		denominator = ev;
// 		v = u/length(u);
// 		u = gradient_hat_x.x*v.x + gradient_hat_x.y*v.y + gradient_hat_x.z*v.z;
// 		ev = dot(v, u);
// 	}
// 	if(fabs(ev-denominator)>para->mc_epsilon)
// 	{
// 		ev = 0.0f;
// 	}
// 	ev = fabs(ev);
// 	// step3: 计算implicit function, denominator为临时变量
// 	denominator = (3.5f - ev)/3.1f;
	return length(x-hat_x) - para->mc_vertex_radius ;//* ((ev<0.4f)?(1.0f):(((denominator - 3.0f)*denominator+3.0f)*denominator));
}

void MC::allocateTextures(uint **d_edgeTable, uint **d_triTable, uint **d_numVertsTable)
{
	checkCudaErrors(cudaMalloc((void **)d_edgeTable, 256*sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_edgeTable, (void *)edgeTable, 256*sizeof(uint), cudaMemcpyHostToDevice));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	checkCudaErrors(cudaBindTexture(0, edgeTex, *d_edgeTable, channelDesc));

	checkCudaErrors(cudaMalloc((void **)d_triTable, 256*16*sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_triTable, (void *)triTable, 256*16*sizeof(uint), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, triTex, *d_triTable, channelDesc));

	checkCudaErrors(cudaMalloc((void **)d_numVertsTable, 256*sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_numVertsTable, (void *)numVertsTable, 256*sizeof(uint), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, numVertsTex, *d_numVertsTable, channelDesc));
}

// compute interpolated vertex along an edge
__device__ float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
	float t = (isolevel - f0) / (f1 - f0);
	return lerp(p0, p1, t);
}

// calculate triangle normal
__device__ float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{
	float3 edge0 = *v1 - *v0;
	float3 edge1 = *v2 - *v0;
	// note - it's faster to perform normalization in vertex shader rather than here
	return cross(edge0, edge1);
}

// version that calculates flat surface normal for each triangle
#ifdef OPT_FOR_SPH
 __global__ void generateTriangles(float3 *vertices, float3 *norm, int *compactedVoxelArray, int *numVertsScanned, int3 voxelSize,
 	int activeVoxels, int maxVerts, int *cellStart, float3 *particlePos, int3 SMCellSize, MC_ParameterSet*para)
#else
inline __device__ float sampleVolume(float *volume, int3 pos, int3 MCCellSize);
__global__ void generateTriangles(float3 *vertices, float3 *norm, int *compactedVoxelArray, int *numVertsScanned, float *volume, int3 voxelSize,
	int activeVoxels, int maxVerts, MC_ParameterSet*para)
#endif
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=activeVoxels) return;

	int voxel = compactedVoxelArray[i];

	int3 gridPos;
	gridPos.x = voxel/(voxelSize.y*voxelSize.z);
	gridPos.y = (voxel - gridPos.x*voxelSize.y*voxelSize.z)/voxelSize.z;
	gridPos.z = voxel - (gridPos.x*voxelSize.y+gridPos.y)*voxelSize.z;

	float3 v[8];
	v[0] = make_float3(gridPos);
	v[1] = make_float3(gridPos + make_int3(1, 0, 0));
	v[2] = make_float3(gridPos + make_int3(1, 1, 0));
	v[3] = make_float3(gridPos + make_int3(0, 1, 0));
	v[4] = make_float3(gridPos + make_int3(0, 0, 1));
	v[5] = make_float3(gridPos + make_int3(1, 0, 1));
	v[6] = make_float3(gridPos + make_int3(1, 1, 1));
	v[7] = make_float3(gridPos + make_int3(0, 1, 1));
	for(int j=0; j<8; j++) {v[j]*=para->mc_cell_length;}

	float field[8];
#ifdef OPT_FOR_SPH
	field[0] = implicitFunction(v[0], SMCellSize, cellStart, particlePos, para);
	field[1] = implicitFunction(v[1], SMCellSize, cellStart, particlePos, para);
	field[2] = implicitFunction(v[2], SMCellSize, cellStart, particlePos, para);
	field[3] = implicitFunction(v[3], SMCellSize, cellStart, particlePos, para);
	field[4] = implicitFunction(v[4], SMCellSize, cellStart, particlePos, para);
	field[5] = implicitFunction(v[5], SMCellSize, cellStart, particlePos, para);
	field[6] = implicitFunction(v[6], SMCellSize, cellStart, particlePos, para);
	field[7] = implicitFunction(v[7], SMCellSize, cellStart, particlePos, para);
#else
	field[0] = sampleVolume(volume, gridPos, voxelSize);
	field[1] = sampleVolume(volume, gridPos + make_int3(1, 0, 0), voxelSize);
	field[2] = sampleVolume(volume, gridPos + make_int3(1, 1, 0), voxelSize);
	field[3] = sampleVolume(volume, gridPos + make_int3(0, 1, 0), voxelSize);
	field[4] = sampleVolume(volume, gridPos + make_int3(0, 0, 1), voxelSize);
	field[5] = sampleVolume(volume, gridPos + make_int3(1, 0, 1), voxelSize);
	field[6] = sampleVolume(volume, gridPos + make_int3(1, 1, 1), voxelSize);
	field[7] = sampleVolume(volume, gridPos + make_int3(0, 1, 1), voxelSize);
#endif

	// recalculate flag
	uint cubeindex;
	cubeindex =  uint(field[0] < para->mc_iso_value);
	cubeindex += uint(field[1] < para->mc_iso_value)*2;
	cubeindex += uint(field[2] < para->mc_iso_value)*4;
	cubeindex += uint(field[3] < para->mc_iso_value)*8;
	cubeindex += uint(field[4] < para->mc_iso_value)*16;
	cubeindex += uint(field[5] < para->mc_iso_value)*32;
	cubeindex += uint(field[6] < para->mc_iso_value)*64;
	cubeindex += uint(field[7] < para->mc_iso_value)*128;

	// find the vertices where the surface intersects the cube
	float3 vertlist[12];

	vertlist[0] = vertexInterp(para->mc_iso_value, v[0], v[1], field[0], field[1]);
	vertlist[1] = vertexInterp(para->mc_iso_value, v[1], v[2], field[1], field[2]);
	vertlist[2] = vertexInterp(para->mc_iso_value, v[2], v[3], field[2], field[3]);
	vertlist[3] = vertexInterp(para->mc_iso_value, v[3], v[0], field[3], field[0]);

	vertlist[4] = vertexInterp(para->mc_iso_value, v[4], v[5], field[4], field[5]);
	vertlist[5] = vertexInterp(para->mc_iso_value, v[5], v[6], field[5], field[6]);
	vertlist[6] = vertexInterp(para->mc_iso_value, v[6], v[7], field[6], field[7]);
	vertlist[7] = vertexInterp(para->mc_iso_value, v[7], v[4], field[7], field[4]);

	vertlist[8] = vertexInterp(para->mc_iso_value, v[0], v[4], field[0], field[4]);
	vertlist[9] = vertexInterp(para->mc_iso_value, v[1], v[5], field[1], field[5]);
	vertlist[10] = vertexInterp(para->mc_iso_value, v[2], v[6], field[2], field[6]);
	vertlist[11] = vertexInterp(para->mc_iso_value, v[3], v[7], field[3], field[7]);

	// output triangle vertices
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

	for(int j=0; j<numVerts; j+=3)
	{
#ifdef OPT_FOR_SPH
		uint index = numVertsScanned[i] + j;
#else
		uint index = numVertsScanned[voxel] + j;
#endif
		float3 *v[3];
		uint edge;
		edge = tex1Dfetch(triTex, (cubeindex*16) + j);
		v[0] = &vertlist[edge];

		edge = tex1Dfetch(triTex, (cubeindex*16) + j + 1);
		v[1] = &vertlist[edge];

		edge = tex1Dfetch(triTex, (cubeindex*16) + j + 2);
		v[2] = &vertlist[edge];

		// calculate triangle surface normal
		float3 n = calcNormal(v[0], v[1], v[2]);

		if(index < (maxVerts - 3))
		{
			vertices[index] = *v[0];
			norm[index] = n;

			vertices[index+1] = *v[1];
			norm[index+1] = n;

			vertices[index+2] = *v[2];
			norm[index+2] = n;
		}
	}
}

void MC::launch_generateTriangles()
{
#ifdef OPT_FOR_SPH
	int gridsize = (num_surfaceCube-1) / BLOCK_SIZE +1;
	generateTriangles<<<gridsize, BLOCK_SIZE>>>(vertices, norm, cubeList2, voxelVerts, MCCellSize, 
		num_surfaceCube, 3*para_h->max_num_triangle, cellStart, particlePos, SMCellSize, para);
#else
	int gridsize = (activeVoxels-1) / BLOCK_SIZE +1;
	gridsize = min(gridsize, GRID_STRIDE);
	generateTriangles<<<gridsize, BLOCK_SIZE>>>(vertices, norm, compactedVoxelArray, voxelVertsScan, volume, MCCellSize, activeVoxels, 3*para_h->max_num_triangle, para);
#endif
cudaDeviceSynchronize(); CHECK_KERNEL();
}


#ifdef OPT_FOR_SPH
__global__ void extractActiveCubes_CUDA(int *cubeMask, int3 cellSize, float3 *pos, int num, MC_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	int cellID;
	__syncthreads();
	int t = para->mc_smoothing_radius_by_cell_length;
#pragma unroll
	for(int m=0; m<(2*t+1)*(2*t+1)*(2*t+1); __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(pos[i]/para->mc_cell_length)+make_int3(m/((2*t+1)*(2*t+1))-t, (m%((2*t+1)*(2*t+1)))/(2*t+1)-t, m%(2*t+1)-t), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		cubeMask[cellID] = 1;
	}
	return ;
}

void MC::extractActiveCubes()
{
	int gridsize = (num_particle-1)/BLOCK_SIZE+1;
	cudaMemset(cubeMask, 0, sizeof(int)*(MCCellSize.x*MCCellSize.y*MCCellSize.z+1));
	extractActiveCubes_CUDA<<<gridsize, BLOCK_SIZE>>>(cubeMask, MCCellSize, particlePos, num_particle, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();

	thrust::exclusive_scan(thrust::device, cubeMask, cubeMask+MCCellSize.x*MCCellSize.y*MCCellSize.z+1, cubeMask);
	cudaMemcpy(&num_activeCube, cubeMask+MCCellSize.x*MCCellSize.y*MCCellSize.z, sizeof(int), cudaMemcpyDeviceToHost);
	printf("MC: activeCubes = %d\n", num_activeCube);

	thrust::pair<int*, int*> new_end;
	thrust::sequence(thrust::device, cubeList1, cubeList1+MCCellSize.x*MCCellSize.y*MCCellSize.z+1, -1, 1);
	new_end = thrust::unique_by_key(thrust::device, cubeMask, cubeMask+MCCellSize.x*MCCellSize.y*MCCellSize.z+1, cubeList1);
	if(new_end.first-cubeMask-1!=num_activeCube)	printf("Marching Cube Error at %s:%d\n", __FILE__, __LINE__);
	cudaMemcpy(cubeList2, cubeList1+1, sizeof(int)*num_activeCube, cudaMemcpyDeviceToDevice);
	return;
}

__global__ void extractSurfaceCubes_CUDA(int *cubeMask, int *voxelVerts, int *cubeList, int3 MCCellSize, int3 SMCellSize, int num, int *cellStart, float3 *particlePos, MC_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	float field[8];
	int3 cellPos;
	uint cubeindex;
	cellPos.x = cubeList[i]/(MCCellSize.z*MCCellSize.y);
	cellPos.y = (cubeList[i] - cellPos.x*MCCellSize.z*MCCellSize.y)/MCCellSize.z;
	cellPos.z = cubeList[i] - (cellPos.x*MCCellSize.y+cellPos.y)*MCCellSize.z;
	__syncthreads();

	field[0] = implicitFunction((make_float3(cellPos) + make_float3(0, 0, 0))*para->mc_cell_length, SMCellSize, cellStart, particlePos, para);
	field[1] = implicitFunction((make_float3(cellPos) + make_float3(1, 0, 0))*para->mc_cell_length, SMCellSize, cellStart, particlePos, para);
	field[2] = implicitFunction((make_float3(cellPos) + make_float3(1, 1, 0))*para->mc_cell_length, SMCellSize, cellStart, particlePos, para);
	field[3] = implicitFunction((make_float3(cellPos) + make_float3(0, 1, 0))*para->mc_cell_length, SMCellSize, cellStart, particlePos, para);
	field[4] = implicitFunction((make_float3(cellPos) + make_float3(0, 0, 1))*para->mc_cell_length, SMCellSize, cellStart, particlePos, para);
	field[5] = implicitFunction((make_float3(cellPos) + make_float3(1, 0, 1))*para->mc_cell_length, SMCellSize, cellStart, particlePos, para);
	field[6] = implicitFunction((make_float3(cellPos) + make_float3(1, 1, 1))*para->mc_cell_length, SMCellSize, cellStart, particlePos, para);
	field[7] = implicitFunction((make_float3(cellPos) + make_float3(0, 1, 1))*para->mc_cell_length, SMCellSize, cellStart, particlePos, para);

	cubeindex =  uint(field[0] < para->mc_iso_value);
	cubeindex += uint(field[1] < para->mc_iso_value)*2;
	cubeindex += uint(field[2] < para->mc_iso_value)*4;
	cubeindex += uint(field[3] < para->mc_iso_value)*8;
	cubeindex += uint(field[4] < para->mc_iso_value)*16;
	cubeindex += uint(field[5] < para->mc_iso_value)*32;
	cubeindex += uint(field[6] < para->mc_iso_value)*64;
	cubeindex += uint(field[7] < para->mc_iso_value)*128;

	// read number of vertices from texture
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);
	voxelVerts[i] = numVerts;
	cubeMask[i] = (numVerts > 0);
	return ;
}

void MC::extractSurfaceCubes()
{
	int gridsize = (num_activeCube-1)/BLOCK_SIZE+1;
	cudaMemset(cubeMask, 0, sizeof(int)*(num_activeCube+1));
	extractSurfaceCubes_CUDA<<<gridsize, BLOCK_SIZE>>>(cubeMask, voxelVerts, cubeList2, MCCellSize, SMCellSize, num_activeCube, cellStart, particlePos, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();

	thrust::exclusive_scan(thrust::device, cubeMask, cubeMask+num_activeCube+1, cubeMask);
	cudaMemcpy(&num_surfaceCube, cubeMask+num_activeCube, sizeof(int), cudaMemcpyDeviceToHost);
	printf("MC: surfaceCubes = %d\n", num_surfaceCube);

	//用cubeList1做临时变量代替cubeMask
	cudaMemcpy(cubeList1, cubeMask, sizeof(int)*(num_activeCube+1), cudaMemcpyDeviceToDevice);
	thrust::pair<int*, int*> new_end;
	new_end = thrust::unique_by_key(thrust::device, cubeList1+1, cubeList1+num_activeCube+1, cubeList2);
	if(new_end.first-cubeList1-2!=num_surfaceCube)	printf("Marching Cube Error at %s:%d\n", __FILE__, __LINE__);

	new_end = thrust::unique_by_key(thrust::device, cubeMask+1, cubeMask+num_activeCube+1, voxelVerts);
	if(new_end.first-cubeMask-2!=num_surfaceCube)	printf("Marching Cube Error at %s:%d\n", __FILE__, __LINE__);
	thrust::exclusive_scan(thrust::device, voxelVerts, voxelVerts+num_surfaceCube+1, voxelVerts);
	return;
}
#else
__global__ void updateField_CUDA(float *volume, int3 MCCellSize, int3 SMCellSize, int num, int *cellStart, float3 *particlePos, MC_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	int x, y, z;
	for(; i<num; __syncthreads(), i+=GRID_STRIDE)
	{
		x = i/((MCCellSize.y+1)*(MCCellSize.z+1));
		y = (i - x*(MCCellSize.y+1)*(MCCellSize.z+1))/(MCCellSize.z+1);
		z = i - (x*(MCCellSize.y+1)+y)*(MCCellSize.z+1);
		volume[i] = implicitFunction(make_float3(x, y, z)*para->mc_cell_length, SMCellSize, cellStart, particlePos, para);
	}

	return ;
}

void MC::updateField()
{
	int gridsize = ((MCCellSize.x+1)*(MCCellSize.y+1)*(MCCellSize.z+1)-1) / BLOCK_SIZE +1;
	gridsize = min(gridsize, GRID_STRIDE);
	updateField_CUDA<<<gridsize, BLOCK_SIZE>>>(volume, MCCellSize, SMCellSize, (MCCellSize.x+1)*(MCCellSize.y+1)*(MCCellSize.z+1), cellStart, particlePos, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();
	return;
}

inline __device__ float sampleVolume(float *volume, int3 pos, int3 MCCellSize)
{
	return volume[(pos.x*(MCCellSize.y+1)+pos.y)*(MCCellSize.z+1)+pos.z];
}

// classify voxel based on number of vertices it will generate
// one thread per voxel
__global__ void classifyVoxel(int *voxelVerts, int *voxelOccupied, float *volume, int3 MCCellSize, MC_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=MCCellSize.x*MCCellSize.y*MCCellSize.z) return;
	int3 gridPos;
	float field[8];
	uint cubeindex;
	for(; i<MCCellSize.x*MCCellSize.y*MCCellSize.z; i+=GRID_STRIDE)
	{
		gridPos.x = i/(MCCellSize.y*MCCellSize.z);
		gridPos.y = (i - gridPos.x*MCCellSize.y*MCCellSize.z)/MCCellSize.z;
		gridPos.z = i - (gridPos.x*MCCellSize.y+gridPos.y)*MCCellSize.z;

		// read field values at neighbouring grid vertices
		field[0] = sampleVolume(volume, gridPos, MCCellSize);
		field[1] = sampleVolume(volume, gridPos + make_int3(1, 0, 0), MCCellSize);
		field[2] = sampleVolume(volume, gridPos + make_int3(1, 1, 0), MCCellSize);
		field[3] = sampleVolume(volume, gridPos + make_int3(0, 1, 0), MCCellSize);
		field[4] = sampleVolume(volume, gridPos + make_int3(0, 0, 1), MCCellSize);
		field[5] = sampleVolume(volume, gridPos + make_int3(1, 0, 1), MCCellSize);
		field[6] = sampleVolume(volume, gridPos + make_int3(1, 1, 1), MCCellSize);
		field[7] = sampleVolume(volume, gridPos + make_int3(0, 1, 1), MCCellSize);

		// calculate flag indicating if each vertex is inside or outside isosurface
		cubeindex =  uint(field[0] < para->mc_iso_value);
		cubeindex += uint(field[1] < para->mc_iso_value)*2;
		cubeindex += uint(field[2] < para->mc_iso_value)*4;
		cubeindex += uint(field[3] < para->mc_iso_value)*8;
		cubeindex += uint(field[4] < para->mc_iso_value)*16;
		cubeindex += uint(field[5] < para->mc_iso_value)*32;
		cubeindex += uint(field[6] < para->mc_iso_value)*64;
		cubeindex += uint(field[7] < para->mc_iso_value)*128;

		// read number of vertices from texture
		uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

		voxelVerts[i] = numVerts;
		voxelOccupied[i] = (numVerts > 0);
	}
}

void MC::launch_classifyVoxel()
{
	int gridsize = (MCCellSize.x*MCCellSize.y*MCCellSize.z-1) / BLOCK_SIZE +1;
	gridsize = min(gridsize, GRID_STRIDE);
	classifyVoxel<<<gridsize, BLOCK_SIZE>>>(voxelVerts, voxelOccupied, volume, MCCellSize, para);
	cudaDeviceSynchronize(); CHECK_KERNEL();
}

void MC::ThrustScanWrapper(int *output, int *input, int numElements)
{
	thrust::exclusive_scan(thrust::device,
		input,
		input + numElements,
		output);
}

__global__ void compactVoxels(int *compactedVoxelArray, int *voxelOccupied, int *voxelOccupiedScan, int numVoxels)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=numVoxels) return;

	for(; i<numVoxels; i+=GRID_STRIDE)
		if(voxelOccupied[i])	compactedVoxelArray[voxelOccupiedScan[i]] = i;
}

void MC::launch_compactVoxels()
{
	int gridsize = (MCCellSize.x*MCCellSize.y*MCCellSize.z-1) / BLOCK_SIZE +1;
	gridsize = min(gridsize, GRID_STRIDE);
	compactVoxels<<<gridsize, BLOCK_SIZE>>>(compactedVoxelArray, voxelOccupied, voxelOccupiedScan, MCCellSize.x*MCCellSize.y*MCCellSize.z);
	cudaDeviceSynchronize(); CHECK_KERNEL();
}

#endif