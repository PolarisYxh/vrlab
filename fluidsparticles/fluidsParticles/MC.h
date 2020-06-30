#ifndef MC_H
#define MC_H
#include "SPH.h"

#define OPT_FOR_SPH // 取消这个宏,使用的是CUDA Samples里的MC算法; 定义这个宏,则在执行MC前先把与粒子距离较远的网格剔除掉,进行加速

typedef struct
{
	float mc_vertex_radius;//r
	float mc_cell_length;//0.5*r
	int mc_smoothing_radius_by_cell_length;
	float mc_smoothing_radius;//4*r
	int mc_ev_iter;
	float mc_epsilon;
	int max_num_triangle;
	float mc_iso_value;
} MC_ParameterSet;

class MC
{
public:
	SPH *sph;
	int3 MCCellSize;
	int3 SMCellSize;
	float3 *particlePos;
	int num_particle;
	float3 spaceSize;
	int *particles2cells; // 粒子位于平滑核计算网格的位置，而不是MC网格的位置
	int *cellStart;
	int num_activeCube, num_surfaceCube;
	float3 *vertices;
	float3 *vertices_h;
	float3 *norm;
	float3 *normal_h;
	MC_ParameterSet *para, *para_h;
	uint *d_numVertsTable = 0;	//分配纹理用
	uint *d_edgeTable = 0;		//分配纹理用
	uint *d_triTable = 0;		//分配纹理用
	int *voxelVerts;	//用来记录每个格子会生成多少顶点
	int totalVerts;

#ifdef OPT_FOR_SPH
	int *cubeList1;
	int *cubeList2;
	int *cubeMask;
	void extractActiveCubes();
	void extractSurfaceCubes();
#else
	int activeVoxels;
	float *volume;
	int *voxelOccupied, *voxelOccupiedScan, *voxelVertsScan, *compactedVoxelArray;
	void updateField();
	void launch_classifyVoxel();
	void ThrustScanWrapper(int *output, int *input, int numElements);
	void launch_compactVoxels();
#endif


	MC(SPH *sph);
	void apply_MC(char *filename);
	void mapParticles2Cells();
	void allocateTextures(uint **d_edgeTable, uint **d_triTable, uint **d_numVertsTable);
	void launch_generateTriangles();
	void output_stl(char * filename);
};

#endif