
#define M_PI 3.14159265358979323846f
#define M_E  2.71828182845904523536f

// cuda
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { printf("CUDA Error at %s:%d\t Error code = %d\n",__FILE__,__LINE__,x);}} while(0) 
#define CHECK_KERNEL(); 	{cudaError_t err = cudaGetLastError();if(err)printf("CUDA Error at %s:%d:\t%s\n",__FILE__,__LINE__,cudaGetErrorString(err));}
#define BLOCK_SIZE 128
#define GRID_STRIDE 65535

template< typename T1, typename T2 >
struct saxpb_functor
{
	const T1 a;
	const T2 b;
	saxpb_functor(T1 _a, T2 _b): a(_a), b(_b) {}
	__host__ __device__
		T2 operator()(const T2& x) const {
		return a * x + b;
	}
};
template< typename T1, typename T2 >
struct saxpy_functor
{
	const T1 a;
	saxpy_functor(T1 _a): a(_a) {}
	__host__ __device__
		T2 operator()(const T2& x, const T2&y) const {
		return a * x + y;
	}
};
template< typename T1, typename T2 >
struct saxpby_functor
{
	const T1 a, b;
	saxpby_functor(T1 _a, T1 _b): a(_a), b(_b) {}
	__host__ __device__
		T2 operator()(const T2& x, const T2&y) const {
		return a * x + b*y;
	}
};

// clock
#define tic() {	clock_t start, stop; start = clock(); 
#define toc(x) stop = clock(); sprintf(frame_log.str[frame_log.ptr++], "%s\t%f", x, (float)(stop - start) / CLOCKS_PER_SEC);}

// logger
struct log
{
	char str[32][128];
	int ptr = 0;
	void clear()
	{
		ptr = 0;
	}
	void output()
	{
		for (int i = 0; i < ptr; i++)
			printf("%s\n", str[i]);
		this->clear();
	}
};