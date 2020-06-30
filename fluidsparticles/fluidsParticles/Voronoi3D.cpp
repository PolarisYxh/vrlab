#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <time.h>
#include "global.h"
#include "SPH.h"
#include "Voronoi3D.h"

Voronoi3D::Voronoi3D(SPH *sph)
{
	this->sph = sph;
	num_site = sph->num_fluid;
	spaceSize = sph->spaceSize;

	para_h=(VORO_ParameterSet*)malloc(sizeof(VORO_ParameterSet));
	para_h->voro_max_spacing = 1.8f*sph->para_h->sph_spacing;
	para_h->voro_cell_length = 1.01f*para_h->voro_max_spacing;
	para_h->voro_epsilon = sph->para_h->sph_epsilon;
	para_h->voro_max_edge = 12*num_site;
	CUDA_CALL(cudaMalloc((void**)&para, sizeof(VORO_ParameterSet)));
	cudaMemcpy(para, para_h, sizeof(VORO_ParameterSet), cudaMemcpyHostToDevice);

	cellSize = make_int3(ceil(spaceSize.x / para_h->voro_cell_length), ceil(spaceSize.y / para_h->voro_cell_length), ceil(spaceSize.z / para_h->voro_cell_length));
	CUDA_CALL(cudaMalloc((void**)&site, sizeof(float3)*num_site));
	CUDA_CALL(cudaMalloc((void**)&particles2cells, sizeof(int)*num_site));
	CUDA_CALL(cudaMalloc((void**)&cellStart, sizeof(int)*(cellSize.x*cellSize.y*cellSize.z+1)));
	CUDA_CALL(cudaMalloc((void**)&num_edge, sizeof(int)));
	CUDA_CALL(cudaMalloc((void**)&edge_flag, sizeof(int)*para_h->voro_max_edge));
	CUDA_CALL(cudaMalloc((void**)&edge, sizeof(int3)*para_h->voro_max_edge));
	CUDA_CALL(cudaMalloc((void**)&circumcircle, sizeof(float3)*para_h->voro_max_edge));

}

