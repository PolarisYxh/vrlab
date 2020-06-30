#ifndef SPH_H
#define SPH_H

#define DFSPH //定义这个宏则使用DFSPH算法([2015][SCA][Divergence-Free Smoothed Particle Hydrodynamics]),取消它则是经典的SPH

typedef struct { float3 x,y,z;} square3; // 粗粒度是列，细粒度是行，如square3.x.y为第二行第一列
typedef struct 
{
	float sph_spacing;
	float sph_smoothing_radius;
	float sph_cell_length;
	float sph_dt;
	float sph_rho0;
	float sph_rhob;
	float sph_m0;
	float sph_k;
	float3 sph_g;
	float xsph_e;
	float visc;
	float sph_boundary_friction;
	float sph_sigma;
	float sph_p_atm;
	float sph_divergence_avg_threshold;
	float sph_density_avg_threshold;
	float sph_epsilon;
	int sph_max_iter;
	int sph_svd_iter;
	float sph_color_energy_coefficient;
} SPH_ParameterSet;

class SPH
{
public:
	SPH_ParameterSet *para, *para_h;

	int num_fluid;
	int num_staticBoundary;
	float3 spaceSize;

	float3 *pos_fluid;
	float3 *vel_fluid;
	float *mass_fluid;
	float3 *pos_staticBoundary;
	float *mass_staticBoundary;
	
	float *density;
	float *pressure;
	int *particles2cells_fluid;
	int *particles2cells2_fluid;
	int *particles2cells_staticBoundary;
	int *cellStart_fluid;
	int *cellStart_staticBoundary;
	float3 *color_grad;
	int3 cellSize;
	int3 compactSize;//边界粒子cube的尺寸
	float *alpha;
	float *error;
	float *stiff_den;
	float *stiff_div;

	float3 *mean_pos;
	square3 *covariance; 
	float3 *singular;
	float *probableSurface;

	SPH(int particleNum, float3 spaceSize);

	void init_fluid();
	void step();
	void neighborSearch();
	void moveParticles();
	void surfaceDetection();

#ifdef DFSPH
	void computeDensityAlpha();
	void applyNonPressureForce();
	void correctDensityError();
	void correctDivergenceError();
#else
	void computeDensity();
	void applyForce();
#endif
	void calculateEnergy();
	void calculateAverageDensity();
	float *buffer_float;
	float3 *buffer_float3;
	bool outputSTL;
};

#endif