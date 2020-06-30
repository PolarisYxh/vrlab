#ifndef VORO_H
#define VORO_H

typedef struct
{
	float voro_max_spacing;
	float voro_cell_length;
	float voro_epsilon;
	int voro_max_edge;
} VORO_ParameterSet;

struct is_zero
{
	__host__ __device__
	bool operator()(const int x)
	{
		return (x==0);
	}
};

class Voronoi3D
{
public:
	Voronoi3D(SPH *sph);
	void DT(); // Delaunay Triangulation

	SPH *sph;
	VORO_ParameterSet *para_h, *para;
	int num_site;
	float3 *site;
	float3 spaceSize;
	int *particles2cells;
	int *cellStart;
	int3 cellSize;

	int *num_edge, num_edge_h;
	int *edge_flag; // valid edge is 1, else 0
	int3 *edge;	// edge.x, edge.y 记录两个端点的索引（x<y)，edge.z是edge数组自己的索引
	float3 *circumcircle;
};

#endif