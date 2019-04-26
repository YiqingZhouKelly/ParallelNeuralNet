
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 8
#include <mxnet/base.h>
/* Note:
 * Data set1: B =10000, M=6, W=48, H=48, K=5, C=1
 * Data set2: B =10000, M=16, W=22, H=22, K=5, C= 6
 */
namespace mxnet
{
namespace op
{
// __constant__ float k_const[2400];
__constant__ float unrolled_k_const[2400];


__global__ void unrollx_kernel(int C, int K,int H, int W, const float* x, float* unroll_x){
    /* DimGrid(ceil(W*1.0/TILE_WIDTH), ceil(H*1.0/TILE_WIDTH), B*C)
     * DimBlock(TILE_WIDTH,TILE_WIDTH,1)
     * threads are mapped to input array x (to reduce mem read)
     * each thread do global memread only once, but do global write at most K*K times.
     */
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    int H_out = H-K+1; 
    int W_out = W-K+1;
    #define unrolledx3d(b,y,x) unroll_x[(b)*(H_out*W_out*C*K*K)+(y)*(W_out*H_out)+(x)]
    int b = blockIdx.z/C;
    int c = blockIdx.z%C;

    int posx = blockIdx.x*TILE_WIDTH+threadIdx.x; 
    int posy = blockIdx.y*TILE_WIDTH+threadIdx.y; //(posy,posx) = location in input array x

    if(posy<H && posx<W){
        int in = x4d(b,c,posy,posx);
        for(int p=0; p<K; ++p){
            for(int q=0; q<K; ++q){ // (p,q) = location in the filter
                int filterStartx= posx-q;
                int filterStarty= posy-p;
                if(filterStarty>=0 && filterStartx>=0){
                    int unroll_col = filterStarty*W_out+ filterStartx;
                    int unroll_row = c*K*K+p*K+q;
                    unrolledx3d(b,unroll_row,unroll_col) = in;
                }
            }
        }
    }
    #undef unrolledx3d
    #undef x4d
}

__global__ void matrixMultiply(const float *A,  float *unrolled_x, float *y,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns,
                                     int W_out, int H_out) {
    /* Warning: Being lazy, I am abusing notation here. A is a 2d array, unrolled_x and output C are 
     * both 3D arrays!!  numrow/ numcol for 3D array are the 2 least significant dimension. The most 
     * significant dim is B, which is mapped to threadidx.z, so do not need to explicitly take care
     * of it during computation, but you do need to be careful when read/ write to 3D arrays.
     */
  // A is already in constant memory
      const float* B = unrolled_x+ blockIdx.z* (numBColumns*numBRows); // b = blockIdx.z
      // __shared__ float subTileM[32][32];
      __shared__ float subTileN[8][8];
      int bx = blockIdx.x;  int by = blockIdx.y;
      int tx = threadIdx.x; int ty = threadIdx.y;

        // Identify the row and column of the P element to work on
      int Row = by * TILE_WIDTH + ty;
      int Col = bx * TILE_WIDTH + tx;
      float Pvalue = 0;
      
      int Width = numCColumns;
      if (numCRows > numCColumns) Width = numCRows;
      if (numAColumns > Width) Width = numAColumns;
      
      for (int m = 0; m < Width/TILE_WIDTH + 1; ++m) {
        // Collaborative loading of M and N tiles into shared memory
        // if (Row < numARows && (m*TILE_WIDTH+tx) < numAColumns) 
        //   subTileM[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
        // else 
        //   subTileM[ty][tx] = 0;
        if (Col < numBColumns && (m*TILE_WIDTH+ty) < numBRows)
          subTileN[ty][tx] = B[threadIdx.z*numBColumns*numBRows+(m*TILE_WIDTH+ty)*numBColumns+Col];
        else
          subTileN[ty][tx] = 0;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
              Pvalue += /*subTileM[ty][k]*/unrolled_k_const[Row*numAColumns+m*TILE_WIDTH+k] * subTileN[k][tx];
        __syncthreads();
     }  
     //numCRows =M
     #define C4d(b,m,y,x) y[(b)*(numCRows*H_out*W_out) +(m)*(H_out*W_out)+(y)*(W_out)+(x)]
     if (Row<numCRows && Col<numCColumns) {
        // C4d(blockIdx.z,Row,Col/W_out, Col%W_out)=Pvalue;
          y[(blockIdx.z)*(numCRows*H_out*W_out) +(Row)*(H_out*W_out)+(Col/W_out)*(W_out)+(Col%W_out)]=Pvalue;

      }
      // C[blockIdx.z*(numCColumns*numCRows)+Row*numCColumns+Col] = Pvalue; //?? illegal mem access?
     #undef C4d
}



/*__global__ void forward_kernel(float *y, const float *x, const float *k, 
                               const int B, const int M, const int C, 
                               const int H, const int W, const int K)
{   
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define TWK (TILE_WIDTH+K)
#define x_shared3d(c,w,h) x_shared[(c)*(TWK)*(TWK)+(w)*(TWK)+(h)]


    extern __shared__ float x_shared[];
    int b,m,h,w,c,p,q,tx,ty;
    
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int W_grid = ceil(W_out / (TILE_WIDTH*1.0));
    h = (blockIdx.z / W_grid)* TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % W_grid)* TILE_WIDTH + threadIdx.x;
    

    b = blockIdx.x;
    m = blockIdx.y;
    tx = threadIdx.x;
    ty = threadIdx.y;

    for(int c=0; c<C; ++c){
        int tid = ty*TILE_WIDTH+tx;
        while(tid<TWK*TWK){
                x_shared[c*TWK*TWK+tid] = x4d(b,c,h+(tid/TWK)-ty,w+(tid%TWK)-tx);
            tid+=TILE_WIDTH*TILE_WIDTH;
        }
    }
    

    __syncthreads();

    float sum = 0.0;
    if(w<W_out && h<H_out){
        for(c = 0; c<C; ++c){
            for(p=0; p<K; ++p){
                for(q =0; q<K; ++q){
                    sum+= x_shared3d(c,ty+p,tx+q)*k_const[m*C*K*K +c*K*K + p*K + q];
                }
            }
        }
        y4d(b,m,h,w) =sum;
    }
        
#undef y4d
#undef x4d
#undef k4d
#undef x_shared3d
#undef TWK
}*/


template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y,
                         const mshadow::Tensor<gpu, 4, float> &x, 
                         const mshadow::Tensor<gpu, 4, float> &w)
{
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    int W_grid = ceil(W_out / (TILE_WIDTH*1.0));
    int H_grid = ceil(H_out / (TILE_WIDTH*1.0));
    int Z = H_grid * W_grid;

    // cudaMemcpyToSymbol(k_const,w.dptr_, sizeof(float)*C*K*K*M,0);
    //*************************************//
    //cpu unrolling k
    float* k = (float*)malloc(sizeof(float)* M*C*K*K);
    cudaMemcpy(k, w.dptr_, sizeof(float)* M*C*K*K, cudaMemcpyDeviceToHost);
    float* unrolled_k = (float*)malloc(sizeof(float)* M*C*K*K);
    for(int m=0; m<M; ++m){
        for(int c=0;c<C;++c){
            for(int p=0; p<K;++p){
                for(int q=0; q<K;++q){
                    unrolled_k[m*C*K*K+c*K*K+p*K+q]=k4d(m,c,p,q);
                }
            }
        }
    }
    free(k);
    cudaMemcpyToSymbol(unrolled_k_const,unrolled_k, sizeof(float)*C*K*K*M,0);
    free(unrolled_k);
    float* unroll_x;
    cudaMalloc((void**)&unroll_x, sizeof(float) *B*C*K*K*H_out*W_out);

    dim3 DimGrid(ceil(W*1.0/TILE_WIDTH), ceil(H*1.0/TILE_WIDTH), B*C);
    dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,1);
    unrollx_kernel<<<DimGrid, DimBlock>>>(C,K,H,W,x.dptr_,unroll_x);

    dim3 DimGridMM(ceil(H_out*W_out*1.0/TILE_WIDTH),ceil(M*1.0/TILE_WIDTH), B);
    dim3 DimBlockMM(TILE_WIDTH,TILE_WIDTH,1);
    matrixMultiply<<<DimGridMM, DimBlockMM>>>(unrolled_k_const,unroll_x, y.dptr_,
                                               M, C*K*K, 
                                               C*K*K,H_out*W_out,
                                               M,H_out*W_out,
                                               W_out,H_out);
    // //***********************************//
    // dim3 gridDim(B,M,Z);
    // dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    // forward_kernel<<<gridDim,blockDim, 
    //                 sizeof(float)*(C*(TILE_WIDTH+K)*(TILE_WIDTH+K))>>>
    //                 (y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    //**********************
    cudaFree(unroll_x);
    //************************
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    #undef k4d
}




template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y,
             const mshadow::Tensor<gpu, 4, DType> &x,
             const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}


}
}

#endif