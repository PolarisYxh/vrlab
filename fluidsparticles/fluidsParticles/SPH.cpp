#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <time.h>
#include "global.h"
#include "SPH.h"

extern struct log frame_log;

SPH::SPH(int particleNum, float3 spaceSize)
{
	this->num_fluid = particleNum;
	this->spaceSize = spaceSize;
	outputSTL = true;

	para_h=(SPH_ParameterSet*)malloc(sizeof(SPH_ParameterSet));
	para_h->sph_spacing = 0.02f;
	para_h->sph_smoothing_radius = 2.0f*para_h->sph_spacing;
	para_h->sph_cell_length = 1.01f*para_h->sph_smoothing_radius;
	para_h->sph_dt = 0.001f;
	para_h->sph_rho0 = 1.0f;
	para_h->sph_rhob = para_h->sph_rho0*1.4f;	// boundary density 把边界粒子的影响增大一些避免流体粒子穿透边界
	para_h->sph_m0 = 76.596750762082e-6f;
	para_h->sph_k = 10.0f;
	para_h->sph_g = make_float3(0.0f, -9.8f, 0.0f);
	para_h->xsph_e = 0.3f;
	para_h->visc = 2e-3f;
	para_h->sph_boundary_friction = 0.5f;
	para_h->sph_sigma = 0.00001f;
	para_h->sph_p_atm = 0.01f;
	para_h->sph_divergence_avg_threshold = 0.001f;
	para_h->sph_density_avg_threshold = 0.0001f;
	para_h->sph_epsilon = 1e-6f;
	para_h->sph_max_iter = 50;
	para_h->sph_svd_iter = 5;
	para_h->sph_color_energy_coefficient = 0.001f;//0.048f;
	CUDA_CALL(cudaMalloc((void**)&para, sizeof(SPH_ParameterSet)));
	cudaMemcpy(para, para_h, sizeof(SPH_ParameterSet), cudaMemcpyHostToDevice);

	cellSize = make_int3(ceil(spaceSize.x / para_h->sph_cell_length), ceil(spaceSize.y / para_h->sph_cell_length), ceil(spaceSize.z / para_h->sph_cell_length));
	compactSize = 2*make_int3(ceil(spaceSize.x / para_h->sph_spacing), ceil(spaceSize.y / para_h->sph_spacing), ceil(spaceSize.z / para_h->sph_spacing));
	this->num_staticBoundary = 
		+compactSize.x*compactSize.y*compactSize.z
		-(compactSize.x-2)*(compactSize.y-2)*(compactSize.z-2);

	CUDA_CALL(cudaMalloc((void**)&pos_fluid, sizeof(float3)*num_fluid));
	CUDA_CALL(cudaMalloc((void**)&vel_fluid, sizeof(float3)*num_fluid));
	CUDA_CALL(cudaMalloc((void**)&mass_fluid, sizeof(float)*num_fluid));
	CUDA_CALL(cudaMalloc((void**)&pos_staticBoundary, sizeof(float3)*num_staticBoundary));
	CUDA_CALL(cudaMalloc((void**)&mass_staticBoundary, sizeof(float)*num_staticBoundary));

	CUDA_CALL(cudaMalloc((void**)&density, sizeof(float)*num_fluid));
	CUDA_CALL(cudaMalloc((void**)&pressure, sizeof(float)*num_fluid));
	CUDA_CALL(cudaMalloc((void**)&particles2cells_fluid, sizeof(int)*num_fluid));
	CUDA_CALL(cudaMalloc((void**)&particles2cells2_fluid, sizeof(int)*num_fluid));
	CUDA_CALL(cudaMalloc((void**)&particles2cells_staticBoundary, sizeof(int)*num_staticBoundary));
	CUDA_CALL(cudaMalloc((void**)&cellStart_fluid, sizeof(int)*(cellSize.x*cellSize.y*cellSize.z+1)));
	CUDA_CALL(cudaMalloc((void**)&cellStart_staticBoundary, sizeof(int)*(cellSize.x*cellSize.y*cellSize.z+1)));

	CUDA_CALL(cudaMalloc((void**)&color_grad, sizeof(float3)*num_fluid));
	CUDA_CALL(cudaMalloc((void**)&alpha, sizeof(float)*(num_fluid)));
	CUDA_CALL(cudaMalloc((void**)&error, sizeof(float)*(num_fluid)));
	CUDA_CALL(cudaMalloc((void**)&stiff_den, sizeof(float)*(num_fluid)));
	CUDA_CALL(cudaMalloc((void**)&stiff_div, sizeof(float)*(num_fluid)));

	CUDA_CALL(cudaMalloc((void**)&mean_pos, sizeof(float3)*num_fluid));
	CUDA_CALL(cudaMalloc((void**)&covariance, sizeof(square3)*num_fluid));
	CUDA_CALL(cudaMalloc((void**)&singular, sizeof(float3)*num_fluid));
	CUDA_CALL(cudaMalloc((void**)&probableSurface, sizeof(float)*num_fluid));

	CUDA_CALL(cudaMalloc((void**)&buffer_float, sizeof(float)*num_fluid));
	CUDA_CALL(cudaMalloc((void**)&buffer_float3, sizeof(float3)*num_fluid));
	init_fluid();
}

void SPH::step()
{
	tic();
#ifdef DFSPH
	tic();
	applyNonPressureForce();
	toc("force");
	tic();
	correctDensityError();
	toc("den");
	calculateEnergy();
	moveParticles();
	neighborSearch();
	tic();
	computeDensityAlpha();
	toc("alpha");
	calculateAverageDensity();
	tic();
	surfaceDetection();
	toc("surface");
	tic();
	correctDivergenceError();
	toc("vel");
#else
	neighborSearch();
	computeDensity();
	calculateAverageDensity();
	applyForce();
	calculateEnergy();
	moveParticles();
#endif
	toc("total");
	return ;
}