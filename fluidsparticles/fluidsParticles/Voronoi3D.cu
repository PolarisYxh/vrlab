#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <helper_math.h>
#include <time.h>
#include "global.h"
#include "SPH.h"
#include "Voronoi3D.h"

extern struct log frame_log;

inline __device__ int particlePos2cellIdx(int3 pos, int3 cellSize)
{
	// return (cellSize.x*cellSize.y*cellSize.z) if the particle is out of the grid
	return (pos.x>=0 && pos.x<cellSize.x && pos.y>=0 && pos.y<cellSize.y && pos.z>=0 && pos.z<cellSize.z)?
		(((pos.x*cellSize.y)+pos.y)*cellSize.z+pos.z)
		: (cellSize.x*cellSize.y*cellSize.z);
}

__global__ void mapParticles2cells_CUDA(int *particles2cells, int3 cellSize, float3 *pos, int num, VORO_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num) return;
	particles2cells[i] = particlePos2cellIdx(make_int3(pos[i]/para->voro_cell_length), cellSize);
	return ;
}

extern "C" __global__ void countingInCell_CUDA(int* cellStart, int *particles2cells, int num);

__global__ void collectEdge_CUDA(int3 *edge, float3 *circumcircle, int *num_edge, float3 *site, int num_site, int *cellStart, int3 cellSize, VORO_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num_site) return;
	int cellID;
#pragma unroll
	for(int m=0; m<27; __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(site[i]/para->voro_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		for (int j=max(i+1, cellStart[cellID]); j<cellStart[cellID+1]; j++) // 保证j<i，这样不会有重复边
		{
			if(length(site[i]-site[j])<para->voro_max_spacing)
			{
				int k = atomicAdd(num_edge, 1);
				edge[k] = make_int3(j,i,k);
				circumcircle[k] = 0.5f*(site[i]+site[j]);
			}
		}
	}
	return ;
}

__device__ float3 get_circumcircle(float3 a1, float3 a2, float3 a3, float3 a4)
{
	float a11, a12, a13, a21, a22, a23, a31, a32, a33, b1, b2, b3, d, d1, d2, d3;
	a11=2.0f*(a2.x-a1.x); a12=2.0f*(a2.y-a1.y); a13=2.0f*(a2.z-a1.z);
	a21=2.0f*(a3.x-a2.x); a22=2.0f*(a3.y-a2.y); a23=2.0f*(a3.z-a2.z);
	a31=2.0f*(a4.x-a3.x); a32=2.0f*(a4.y-a3.y); a33=2.0f*(a4.z-a3.z);
	b1=a2.x*a2.x-a1.x*a1.x+a2.y*a2.y-a1.y*a1.y+a2.z*a2.z-a1.z*a1.z;
	b2=a3.x*a3.x-a2.x*a2.x+a3.y*a3.y-a2.y*a2.y+a3.z*a3.z-a2.z*a2.z;
	b3=a4.x*a4.x-a3.x*a3.x+a4.y*a4.y-a3.y*a3.y+a4.z*a4.z-a3.z*a3.z;
	d=a11*a22*a33+a12*a23*a31+a13*a21*a32-a11*a23*a32-a12*a21*a33-a13*a22*a31;
	d1=b1*a22*a33+a12*a23*b3+a13*b2*a32-b1*a23*a32-a12*b2*a33-a13*a22*b3;
	d2=a11*b2*a33+b1*a23*a31+a13*a21*b3-a11*a23*b3-b1*a21*a33-a13*b2*a31;
	d3=a11*a22*b3+a12*b2*a31+b1*a21*a32-a11*b2*a32-a12*a21*b3-b1*a22*a31;
	return make_float3(d1,d2,d3)/d;
}

__global__ void correctCircumcircle_CUDA(int *edge_flag, int3 *edge, float3 *circumcircle, int num_edge, float3 *site, int *cellStart, int3 cellSize, VORO_ParameterSet *para)
{
	unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(i>=num_edge) return;
	int cellID;
	//int num_neighbor = 0;
	//int neighbor[6]={edge[i].x, edge[i].y, edge[i].y, edge[i].y, edge[i].y, edge[i].y};
	int neighbor[2]= {-1,-1};
	float min_theta = 0.0f;
	//int t;
#pragma unroll
	for(int m=0; m<27; __syncthreads(), m++)
	{
		cellID = particlePos2cellIdx(make_int3(circumcircle[i]/para->voro_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
		if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
		for(int j=cellStart[cellID]; j<cellStart[cellID+1]; j++)
		{
			if(j==edge[i].x||j==edge[i].y) continue;
			float3 a,b;
			a = site[edge[i].x] - site[j]; b = site[edge[i].y] - site[j];
			float cos_theta = dot(a,b)/(length(a)*length(b)); //应该是钝角or直角，非正数
			if(cos_theta<min_theta+para->voro_epsilon)
			{
				min_theta = cos_theta;
				neighbor[0] = j;
			}

// 			num_neighbor++;
// 			neighbor[5] = j;
// 			// 只需一次冒泡
// 			if(length(circumcircle[i] - site[neighbor[4]]) > para->voro_epsilon + length(circumcircle[i] - site[neighbor[5]])) { t=neighbor[5]; neighbor[5] = neighbor[4]; neighbor[4] = t; }
// 			if(length(circumcircle[i] - site[neighbor[3]]) > para->voro_epsilon + length(circumcircle[i] - site[neighbor[4]])) { t=neighbor[4]; neighbor[4] = neighbor[3]; neighbor[3] = t; }
// 			if(length(circumcircle[i] - site[neighbor[2]]) > para->voro_epsilon + length(circumcircle[i] - site[neighbor[3]])) { t=neighbor[3]; neighbor[3] = neighbor[2]; neighbor[2] = t; }
// 			if(length(circumcircle[i] - site[neighbor[1]]) > para->voro_epsilon + length(circumcircle[i] - site[neighbor[2]])) { t=neighbor[2]; neighbor[2] = neighbor[1]; neighbor[1] = t; }
// 			if(length(circumcircle[i] - site[neighbor[0]]) > para->voro_epsilon + length(circumcircle[i] - site[neighbor[1]])) { t=neighbor[1]; neighbor[1] = neighbor[0]; neighbor[0] = t; }
		}
	}

	if(neighbor[0]>=0)	//找到了比边的端点离圆心更近的点
	{
		// 修正圆心
		float a,b,c, R; //三角形边长，外接圆半径
		a = length(site[edge[i].x] - site[neighbor[0]]);
		b = length(site[edge[i].y] - site[neighbor[0]]);
		c = length(site[edge[i].x] - site[edge[i].y]);
		R = a*b*c/sqrtf((a+b+c)*(a+b-c)*(a-b+c)*(-a+b+c));
		if(R>para->voro_max_spacing)
			;//edge_flag[i] = 0; //圆太大了，假设正常粒子间距不会超过voro_max_spacing，所以这条边不是Delaunay边
		else
		{
			float3 n1,n2;
			n1 = normalize(cross(site[edge[i].x] - site[neighbor[0]], site[edge[i].y] - site[neighbor[0]])); // 圆所在平面的法向
			n2 = normalize(site[edge[i].x] - site[edge[i].y]); // 边的方向
			circumcircle[i] += cross(n1,n2) * sqrtf(R*R - 0.25f*c*c);
			min_theta = 0.0f;
			for(int m=0; m<27; __syncthreads(), m++)
			{
				cellID = particlePos2cellIdx(make_int3(circumcircle[i]/para->voro_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
				if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
				for(int j=cellStart[cellID]; j<cellStart[cellID+1]; j++)
				{
					if(j==edge[i].x||j==edge[i].y||j==neighbor[0]) continue;
					// 点投影到平面，垂足为p，到平面上一点的向量为d
					float3 p, d = site[edge[i].x] - site[j];
					p = site[j] + n1 * length(d) * dot(d, n1) / (length(d)*length(n1));
					float3 a, b; //垂足与圆心的连线交圆于两点
					a = circumcircle[i] + R*(p-circumcircle[i])/length(p-circumcircle[i]);
					b = circumcircle[i] - R*(p-circumcircle[i])/length(p-circumcircle[i]);
					a = a - site[j]; b = b - site[j];
					float cos_theta = dot(a, b)/(length(a)*length(b));
					if(cos_theta<min_theta+para->voro_epsilon)
					{
// 						if(dot(normalize(cross(site[j]-site[edge[i].x], site[j]-site[edge[i].y])), normalize(site[j]-site[neighbor[0]]))<0.2f)
// 							if(neighbor[1] <0) 
// 							{
// 								min_theta = cos_theta;
// 								neighbor[1] = neighbor[0];	//避免四点成球那一步时四个点是共面的
// 							}
// 						else
						{
							min_theta = cos_theta;
							neighbor[1] = j;
						}
					}
				}
			}
			if(neighbor[1]>=0)
			{
				// 四点共面
				if( dot(normalize(cross(site[edge[i].x]-site[edge[i].y], site[edge[i].x]-site[neighbor[0]])), normalize(site[edge[i].x] - site[neighbor[1]])) < 0.1f )
					atomicExch(&edge_flag[i], 4);
				// 四点成球
				circumcircle[i] = get_circumcircle(site[edge[i].x], site[edge[i].y], site[neighbor[0]], site[neighbor[1]]);
				R = length(circumcircle[i]-site[edge[i].x]);
				if(R>para->voro_max_spacing || neighbor[0]==neighbor[1])
					;//edge_flag[i] = 0; //球太大了，假设正常粒子间距不会超过voro_max_spacing，所以这条边不是Delaunay边
				else
				{
					int innerCount = 0;
					for(int m=0; m<27; __syncthreads(), m++)
					{
						cellID = particlePos2cellIdx(make_int3(circumcircle[i]/para->voro_cell_length)+make_int3(m/9-1, (m%9)/3-1, m%3-1), cellSize);
						if(cellID==(cellSize.x*cellSize.y*cellSize.z)) continue;
						for(int j=cellStart[cellID]; j<cellStart[cellID+1] && innerCount==0; j++)
						{
							if(j==edge[i].x||j==edge[i].y||j==neighbor[0]||j==neighbor[1]) continue;
							if(length(site[j] - circumcircle[i])<R+para->voro_epsilon) 
							{
								innerCount++;
								break;
							}
						}
					}
					if(innerCount==0)
						atomicMax(&edge_flag[i], 3);//edge_flag[i]=1;
				}
			}
			else
				atomicMax(&edge_flag[i], 2);//edge_flag[i] = 1;
		}
	}
	else
		atomicMax(&edge_flag[i], 1);//edge_flag[i] = 1;

// 	if((neighbor[0]==edge[i].x && neighbor[1]==edge[i].y) || (neighbor[0]==edge[i].y && neighbor[1]==edge[i].x))
// 		edge_flag[i] = 1;
// 	else 
// 	{
// 		float3 c = get_circumcircle(site[neighbor[0]],site[neighbor[1]],site[neighbor[2]],site[neighbor[3]]);
// 		if(length(c-site[neighbor[4]]) > para->voro_epsilon +  length(c-site[neighbor[0]]))
// 			edge_flag[i] = 1;
// 	}
	return ;
}

void Voronoi3D::DT()
{
	tic();

	cudaMemset(num_edge, 0, sizeof(int));
	cudaMemset(edge_flag, 0, sizeof(int)*para_h->voro_max_edge);
	cudaMemcpy(site, sph->pos_fluid, sizeof(float3)*num_site, cudaMemcpyDeviceToDevice);

	int gridsize = (num_site-1)/BLOCK_SIZE+1;
	// neighbor search
	mapParticles2cells_CUDA<<<gridsize, BLOCK_SIZE>>>(particles2cells, cellSize, site, num_site, para);
	thrust::sort_by_key(thrust::device, particles2cells, particles2cells + num_site, site);
	cudaMemset(cellStart, 0, sizeof(int)*(cellSize.x*cellSize.y*cellSize.z+1));
	countingInCell_CUDA<<<gridsize, BLOCK_SIZE>>>(cellStart, particles2cells, num_site);
	thrust::exclusive_scan(thrust::device, cellStart, cellStart+cellSize.x*cellSize.y*cellSize.z+1, cellStart);

	// collect possible edges and predict circumcircle
	collectEdge_CUDA<<<gridsize, BLOCK_SIZE>>>(edge, circumcircle, num_edge, site, num_site, cellStart, cellSize, para);
	cudaMemcpy(&num_edge_h, num_edge, sizeof(int), cudaMemcpyDeviceToHost);

	sprintf(frame_log.str[frame_log.ptr++], "site_num=%d, edge_num=%d", num_site, num_edge_h);

	gridsize = (num_edge_h-1)/BLOCK_SIZE+1;
	// correct circumcircle
	correctCircumcircle_CUDA<<<gridsize, BLOCK_SIZE>>>(edge_flag, edge, circumcircle, num_edge_h, site, cellStart, cellSize, para);
	// reject non-Delaunay edges
	int3* new_end = thrust::remove_if(thrust::device, edge, edge+num_edge_h, edge_flag, is_zero());
	num_edge_h = new_end - edge;
	sprintf(frame_log.str[frame_log.ptr++], "site_num=%d, edge_num=%d", num_site, num_edge_h);

	toc("Delaunay Triangulation");
	return ;
}
