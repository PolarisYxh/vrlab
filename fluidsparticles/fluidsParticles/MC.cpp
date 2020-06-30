#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <time.h>
#include "global.h"
#include "MC.h"

extern struct log frame_log;

MC::MC(SPH *sph)
{
	this->sph = sph;
	num_particle = sph->num_fluid;
	spaceSize = sph->spaceSize;
	para_h=(MC_ParameterSet*)malloc(sizeof(MC_ParameterSet));
	para_h->mc_vertex_radius = 0.5f*sph->para_h->sph_spacing;
	para_h->mc_cell_length = 0.25f*sph->para_h->sph_spacing;
	para_h->mc_smoothing_radius_by_cell_length = 8;
	para_h->mc_smoothing_radius = para_h->mc_smoothing_radius_by_cell_length*para_h->mc_cell_length;
	para_h->mc_ev_iter = 100;
	para_h->mc_epsilon = 1e-6f;
	para_h->max_num_triangle = 10000000;
	para_h->mc_iso_value = 0.0f;
	CUDA_CALL(cudaMalloc((void**)&para, sizeof(MC_ParameterSet)));
	cudaMemcpy(para, para_h, sizeof(MC_ParameterSet), cudaMemcpyHostToDevice);

	MCCellSize = make_int3(ceil(spaceSize.x / para_h->mc_cell_length), ceil(spaceSize.y / para_h->mc_cell_length), ceil(spaceSize.z / para_h->mc_cell_length));
	SMCellSize = make_int3(ceil(spaceSize.x / para_h->mc_smoothing_radius), ceil(spaceSize.y / para_h->mc_smoothing_radius), ceil(spaceSize.z / para_h->mc_smoothing_radius));
	CUDA_CALL(cudaMalloc((void**)&particles2cells, sizeof(int)*num_particle));
#ifdef OPT_FOR_SPH
	CUDA_CALL(cudaMalloc((void**)&cubeList1, sizeof(int)*(MCCellSize.x*MCCellSize.y*MCCellSize.z+1)));
	CUDA_CALL(cudaMalloc((void**)&cubeList2, sizeof(int)*(MCCellSize.x*MCCellSize.y*MCCellSize.z+1)));
	CUDA_CALL(cudaMalloc((void**)&cubeMask, sizeof(int)*(MCCellSize.x*MCCellSize.y*MCCellSize.z+1)));
	cudaMemset(cubeList1, 0, sizeof(int)*(MCCellSize.x*MCCellSize.y*MCCellSize.z+1));
	cudaMemset(cubeList2, 0, sizeof(int)*(MCCellSize.x*MCCellSize.y*MCCellSize.z+1));
	cudaMemset(cubeMask, 0, sizeof(int)*(MCCellSize.x*MCCellSize.y*MCCellSize.z+1));
#else
	CUDA_CALL(cudaMalloc((void**)&volume, sizeof(float)*(MCCellSize.x+1)*(MCCellSize.y+1)*(MCCellSize.z+1)));
	CUDA_CALL(cudaMalloc((void**)&voxelOccupied, sizeof(int)*(MCCellSize.x)*(MCCellSize.y)*(MCCellSize.z)));
	CUDA_CALL(cudaMalloc((void**)&voxelOccupiedScan, sizeof(int)*(MCCellSize.x)*(MCCellSize.y)*(MCCellSize.z)));
	CUDA_CALL(cudaMalloc((void**)&compactedVoxelArray, sizeof(int)*(MCCellSize.x)*(MCCellSize.y)*(MCCellSize.z)));
	CUDA_CALL(cudaMalloc((void**)&voxelVertsScan, sizeof(int)*((MCCellSize.x)*(MCCellSize.y)*(MCCellSize.z)+1)));
#endif
	CUDA_CALL(cudaMalloc((void**)&voxelVerts, sizeof(int)*((MCCellSize.x)*(MCCellSize.y)*(MCCellSize.z)+1)));
 	CUDA_CALL(cudaMalloc((void**)&particlePos, sizeof(float3)*num_particle));
	CUDA_CALL(cudaMalloc((void**)&cellStart, sizeof(int)*(SMCellSize.x*SMCellSize.y*SMCellSize.z+1)));
	CUDA_CALL(cudaMalloc((void**)&vertices, sizeof(float3)*3*para_h->max_num_triangle));
	CUDA_CALL(cudaMalloc((void**)&norm, sizeof(float3)*3*para_h->max_num_triangle));
	vertices_h = (float3 *)malloc(sizeof(float3)*3*para_h->max_num_triangle);
	normal_h = (float3 *)malloc(sizeof(float3)*3*para_h->max_num_triangle);
	allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);
}

void MC::apply_MC(char *filename)
{
	cudaMemcpy(particlePos, sph->pos_fluid, sizeof(float3)*num_particle, cudaMemcpyDeviceToDevice);
	tic();

	mapParticles2Cells();
#ifdef OPT_FOR_SPH
	// step1: 遍历粒子,把粒子附近的MC格子找到,放入activeCubes集合
	extractActiveCubes();
	// step2: 从activeCubes集合中把与isosurface相交的MC格子找到,放入surfaceCubes集合
	extractSurfaceCubes();
	cudaMemcpy((void *)&totalVerts, (void *)(voxelVerts + num_surfaceCube), sizeof(int), cudaMemcpyDeviceToHost);
	// step3: 由surfaceCubes集合生成三角面片
	launch_generateTriangles();
#else
	// step1: 把isovalue存成标量场
	updateField();

	// step2: 检查每个MC格子是否被占用以及会生成多少顶点
	launch_classifyVoxel();

	// scan voxel occupied array
	ThrustScanWrapper(voxelOccupiedScan, voxelOccupied, MCCellSize.x*MCCellSize.y*MCCellSize.z);

	// read back values to calculate total number of non-empty voxels
	// since we are using an exclusive scan, the total is the last value of
	// the scan result plus the last value in the input array
	{
		uint lastElement, lastScanElement;
		cudaMemcpy((void *)&lastElement, (void *)(voxelOccupied + MCCellSize.x*MCCellSize.y*MCCellSize.z-1), sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy((void *)&lastScanElement, (void *)(voxelOccupiedScan + MCCellSize.x*MCCellSize.y*MCCellSize.z-1), sizeof(int), cudaMemcpyDeviceToHost);
		activeVoxels = lastElement + lastScanElement;
	}

	if(activeVoxels!=0)
	{
		// step3: 把与isosurface相交的MC格子找到
		launch_compactVoxels();

		// scan voxel vertex count array
		ThrustScanWrapper(voxelVertsScan, voxelVerts, MCCellSize.x*MCCellSize.y*MCCellSize.z);

		// readback total number of vertices
		{
			uint lastElement, lastScanElement;
			cudaMemcpy((void *)&lastElement, (void *)(voxelVerts + MCCellSize.x*MCCellSize.y*MCCellSize.z-1), sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy((void *)&lastScanElement, (void *)(voxelVertsScan + MCCellSize.x*MCCellSize.y*MCCellSize.z-1), sizeof(int), cudaMemcpyDeviceToHost);
			totalVerts = lastElement + lastScanElement;
		}
		// step4: 生成三角面片
		launch_generateTriangles();
	}
	else
		totalVerts=0;
#endif

	printf("MC: totalVerts = %d\n", totalVerts);
	toc("MC");

	output_stl(filename);
	return;
}

void MC::output_stl(char * filename)
{
	printf("MC: starting output...\n");
	tic();
	int num_triangles = totalVerts/3;
	cudaMemcpy(vertices_h, vertices, sizeof(float3)*3*num_triangles, cudaMemcpyDeviceToHost);
	cudaMemcpy(normal_h, norm, sizeof(float3)*3*num_triangles, cudaMemcpyDeviceToHost);
	FILE *fp = fopen(filename, "w");
	fprintf(fp, "solid Exported from Blender-2.76 (sub 0)\n");
	for(int i=0; i<num_triangles; i++)
	{
		fprintf(fp, "facet normal %f %f %f\n", normal_h[3*i].x, normal_h[3*i].y, normal_h[3*i].z);
		//fprintf(fp, "facet normal %f %f %f\n", 0, 0, 0);
		fprintf(fp, "outer loop\n");
		fprintf(fp, "vertex %f %f %f\n", vertices_h[i*3].x, vertices_h[i*3].z, vertices_h[i*3].y);
		fprintf(fp, "vertex %f %f %f\n", vertices_h[i*3+1].x, vertices_h[i*3+1].z, vertices_h[i*3+1].y);
		fprintf(fp, "vertex %f %f %f\n", vertices_h[i*3+2].x, vertices_h[i*3+2].z, vertices_h[i*3+2].y);
		fprintf(fp, "endloop\n");
		fprintf(fp, "endfacet\n");
	}
	fprintf(fp, "endsolid Exported from Blender-2.76 (sub 0)\n");
	fclose(fp);
	toc("output");
	printf("MC: output done.\n");
	return;
}
