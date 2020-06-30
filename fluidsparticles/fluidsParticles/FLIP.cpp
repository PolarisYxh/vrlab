#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <helper_math.h>
#include <time.h>
#include "global.h"
#include "SPH.h"
#include "FLIP.h"

extern struct log frame_log;

FLIP::FLIP(SPH *sph)
{
	this->sph = sph;
	this->num_flip = 8*sph->num_fluid;
	this->spaceSize = sph->spaceSize;
	this->cellSize = sph->cellSize;

	para_h=(FLIP_ParameterSet*)malloc(sizeof(FLIP_ParameterSet));
	para_h->sph_spacing = sph->para_h->sph_spacing;
	para_h->sph_smoothing_radius = sph->para_h->sph_smoothing_radius;
	para_h->sph_cell_length = sph->para_h->sph_cell_length;
	para_h->sph_dt = sph->para_h->sph_dt;
	para_h->flip_mass = sph->para_h->sph_m0;
	para_h->flip_g = sph->para_h->sph_g;
	para_h->flip_epsilon = sph->para_h->sph_epsilon;
	para_h->pic_alpha = 0.05f;
	para_h->flip_position_correction_gamma = 0.05f;
	CUDA_CALL(cudaMalloc((void**)&para, sizeof(FLIP_ParameterSet)));
	cudaMemcpy(para, para_h, sizeof(FLIP_ParameterSet), cudaMemcpyHostToDevice);

	CUDA_CALL(cudaMalloc((void**)&pos_flip, sizeof(float3)*num_flip));
	CUDA_CALL(cudaMalloc((void**)&pos_correction, sizeof(float3)*num_flip));
	CUDA_CALL(cudaMalloc((void**)&vel_flip, sizeof(float3)*num_flip));
	CUDA_CALL(cudaMalloc((void**)&vel_fluid_last, sizeof(float3)*sph->num_fluid));
	CUDA_CALL(cudaMalloc((void**)&particles2cells, sizeof(int)*num_flip));
	CUDA_CALL(cudaMalloc((void**)&particles2cells2, sizeof(int)*num_flip));
	CUDA_CALL(cudaMalloc((void**)&cellStart, sizeof(int)*(cellSize.x*cellSize.y*cellSize.z+1)));

	init_flip(sph);
}

void FLIP::step()
{
#ifdef DFSPH
	tic();
	sph->applyNonPressureForce();
	sph->correctDensityError();

	// flip process
	// 1st velocity mapping
	tic();
	sph2flipMapping();
	toc("sph2flipMapping");
	// 2nd advect
	tic();
	flipAdvect();
	toc("flipAdvect");

	sph->moveParticles();
	sph->neighborSearch();
	// 3rd velocity mapping
	tic();
	flip2sphMapping();
	cudaMemcpy(vel_fluid_last, sph->vel_fluid, sizeof(float3)*sph->num_fluid, cudaMemcpyDeviceToDevice);
	toc("flip2sphMapping");

	sph->computeDensityAlpha();
	sph->correctDivergenceError();
	toc("total flip");
#endif
	return;
}