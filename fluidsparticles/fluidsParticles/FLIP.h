#ifndef FLIP_H
#define FLIP_H

typedef struct
{
	float sph_spacing;
	float sph_smoothing_radius;
	float sph_cell_length;
	float sph_dt;
	float flip_mass;
	float3 flip_g;
	float flip_epsilon;
	float pic_alpha;
	float flip_position_correction_gamma;
} FLIP_ParameterSet;

class FLIP
{
public:
	FLIP_ParameterSet *para, *para_h;
	SPH *sph;
	int num_flip;
	float3 spaceSize;
	int3 cellSize;
	int *particles2cells;
	int *particles2cells2;
	int *cellStart;

	float3 *pos_flip;
	float3 *pos_correction;
	float3 *vel_flip;
	float3 *vel_fluid_last;

	FLIP(SPH *sph);

	void init_flip(SPH *sph);
	void step();
	void neighborSearch();
	void sph2flipMapping();
	void flipAdvect();
	void flip2sphMapping();
};

#endif