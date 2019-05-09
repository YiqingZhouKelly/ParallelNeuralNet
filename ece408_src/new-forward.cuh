
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 32
#define TILE_WIDTH_SMALL 16
#include <mxnet/base.h>
/* Note:
 * Data set1: B =10000, M=6, W=48, H=48, K=5, C=1
 * Data set2: B =10000, M=16, W=22, H=22, K=5, C= 6
 */
namespace mxnet
{
namespace op
{
__constant__ float k_const[2400];


__global__ void small_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K,int H_out, int W_out)
{   
    __shared__ float x_shared[400];

    int b,m,h,w,p,q;
    // b = blockIdx.x;
    // m = blockIdx.y;
    b = blockIdx.z;
    // const int H_out = H - K + 1;
    // const int W_out = W - K + 1;
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i1, i0) x[(i3) * (C * H * W) + (i1) * (W) + i0]
    #define k4d(i3, i1, i0) k_const[(i3) * (C * K * K) + (i1) * (K) + i0]

    // int W_grid = ceil(W_out / (TILE_WIDTH_SMALL*1.0));
    // h = (blockIdx.z / W_grid)* TILE_WIDTH_SMALL + threadIdx.y;
    // w = (blockIdx.z % W_grid)* TILE_WIDTH_SMALL + threadIdx.x;

    int tid = threadIdx.y*TILE_WIDTH_SMALL +threadIdx.x;
    int tileStartx= blockIdx.x*TILE_WIDTH_SMALL;
    int tileStarty = blockIdx.y*TILE_WIDTH_SMALL;
    while(tid<400){
      int offsetx = tid%20;
      int offsety = tid/20;
      x_shared[tid] = x4d(b,tileStarty+offsety, tileStartx+offsetx);
      tid+=TILE_WIDTH_SMALL*TILE_WIDTH_SMALL;
    }

    __syncthreads();
    w=blockIdx.x*TILE_WIDTH_SMALL+threadIdx.x;
    h=blockIdx.y*TILE_WIDTH_SMALL+threadIdx.y;
    if(w<W_out && h<H_out){
    // for(c = 0; c<C; ++c){ 
    for(m=0; m<M;++m){
      float sum = 0.0;
        for(p=0; p<K; ++p){
            for(q =0; q<K; ++q){
              sum+= /*x4d(b,h+p,w+q)*/ x_shared[20* (threadIdx.y+p)+(threadIdx.x+ q)]*k4d(m,p,q);
            }
        }
    // }
    y4d(b,m,h,w) =sum;
  }
}
    #undef y4d
    #undef x4d
    #undef k4d
}

__global__ void unrollx_kernel(int C, int K,int H, int W, const float* x, float* unroll_x){
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
                if(filterStarty>=0 && filterStartx>=0 && filterStartx<W_out && filterStarty<H_out){
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

__global__ void unrollx_mapout(int C, int K, int H, int W, const float* x, float* unroll_x){
  // int b = blockIdx.z;
  int H_out = H-K+1;
  int W_out = W-K+1;
  int posx = blockIdx.x* TILE_WIDTH+threadIdx.x;
  int posy = blockIdx.y*TILE_WIDTH +threadIdx.y;
  if(posx<H_out*W_out && posy<C*K*K){
    int c = posy/(K*K);
    // int yy = (posy%(K*K))/K + (posx/W_out);
    // int xx = (posy%(K*K))%K + (posx%W_out);
    // int filterStartx=posx%W_out;
    // int filterStarty=posx/W_out;
    // yy+=filterStarty;
    // xx+=filterStartx;
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    unroll_x[blockIdx.z*C*K*K*H_out*W_out+posy*H_out*W_out+posx]= x4d(blockIdx.z,c,(posy%(K*K))/K + (posx/W_out),(posy%(K*K))%K + (posx%W_out));
  }
}

__global__ void matrixMultiply(float *A, float* unrolled_x, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns, int W_out, int H_out) {
  int b = blockIdx.z;
  const float* B = unrolled_x+ blockIdx.z* (numBColumns*numBRows); // b = blockIdx.z
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  if((row<numCRows) && (col<numCColumns)){
    float sum = 0.0;
    for (int k = 0; k < numBRows; k++){
      sum +=k_const[row*numAColumns+k]*B[k*numBColumns+col];
    }
    C[(blockIdx.z)*(numCRows*H_out*W_out) +(row)*(H_out*W_out)+(col/W_out)*(W_out)+(col%W_out)] = sum;
  }
}


__global__ void forward_kernel(int C, int K, int H, int W, int M,const float* x, float*y,int H_out, int W_out){
  //copied from unrollx_mapout 
  __shared__ float tile[150][32];
  int b = blockIdx.z;
  int posx = blockIdx.x* TILE_WIDTH+threadIdx.x;
  int posy = blockIdx.y*TILE_WIDTH +threadIdx.y;
  if(posx<H_out*W_out){
  for(int j = 0; j< ceil(C*K*K*1.0/TILE_WIDTH); ++j){
  if(posy<C*K*K){
    int c = posy/(K*K);
    int yy = (posy%(K*K))/K+posx/W_out;
    int xx = (posy%(K*K))%K+ posx%W_out;
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    tile[posy][threadIdx.x] = x4d(b,c,yy,xx);
    posy+=TILE_WIDTH;
  }
}
  //MM
     posy = blockIdx.y*TILE_WIDTH +threadIdx.y;
      __syncthreads();
        if(posy< M){
          float sum = 0.;
          for(int i=0; i<C*K*K; ++i){
            sum +=k_const[posy*C*K*K+i]* tile[i][threadIdx.x];
          }
          y[b*M*H_out*W_out+posy*H_out*W_out+posx]=sum;
        }
      }
#undef x4d
}




__global__ void specialMM(const float *A,  float *unrolled_x, float *y,
                          int B, int M, int C, int H,  int W, int K){ //input ckkmb

__shared__ float tile[150][32/**2*/]; // case1
int b = blockIdx.z;
int H_out = H-K+1;
int W_out = W-K+1;
int posy = blockIdx.y*TILE_WIDTH+threadIdx.y; //threadidx.y=posy...
int posx = blockIdx.x*TILE_WIDTH/**2*/+threadIdx.x; //position of thread in output
// for(int j=0; j<2; ++j){
  if(posx/*+j*TILE_WIDTH*/<H_out*W_out){
    for(int i=0;i<ceil(C*K*K*1.0/TILE_WIDTH);++i){
      if(threadIdx.y+TILE_WIDTH*i<C*K*K)
        tile[i*TILE_WIDTH+threadIdx.y][threadIdx.x/*+j*TILE_WIDTH*/]= unrolled_x[b*C*K*K*W_out*H_out +(threadIdx.y+TILE_WIDTH*i)*(W_out*H_out)+posx/*+j*TILE_WIDTH*/];
    }
  }
// }
__syncthreads();
// for(int j=0; j<2; ++j){
  if(posx/*+j*TILE_WIDTH*/<H_out*W_out && posy< M){
    float sum = 0.;
    for(int i=0; i<C*K*K; ++i){
      sum +=k_const[posy*C*K*K+i]* tile[i][threadIdx.x/*+j*TILE_WIDTH*/];
    }
    y[b*M*H_out*W_out+posy*H_out*W_out+posx/*+j*TILE_WIDTH*/]=sum;
  }
// }

}

__global__ void matrixMultiplyTiled(const float *A,  float *unrolled_x, float *y,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns,
                                     int W_out, int H_out) {
      const float* B = unrolled_x+ blockIdx.z* (numBColumns*numBRows); 
      __shared__ float subTileN[32][32];
      int bx = blockIdx.x;  int by = blockIdx.y;
      int tx = threadIdx.x; int ty = threadIdx.y;

      int Row = by * TILE_WIDTH + ty;
      int Col = bx * TILE_WIDTH + tx;
      float Pvalue = 0;
      
      int Width = numCColumns;
      if (numCRows > numCColumns) Width = numCRows;
      if (numAColumns > Width) Width = numAColumns;
      
      for (int m = 0; m < Width/TILE_WIDTH + 1; ++m) {
        if (Col < numBColumns && (m*TILE_WIDTH+ty) < numBRows)
          subTileN[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
        else
          subTileN[ty][tx] = 0;
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k)
          if (Row < numARows && (m*TILE_WIDTH+k) < numAColumns)
              Pvalue += k_const[Row*numAColumns+m*TILE_WIDTH+k] * subTileN[k][tx];
        __syncthreads();
     }  
     #define C4d(b,m,y,x) y[(b)*(numCRows*H_out*W_out) +(m)*(H_out*W_out)+(y)*(W_out)+(x)]
     if (Row<numCRows && Col<numCColumns) {
          y[(blockIdx.z)*(numCRows*H_out*W_out) +(Row)*(H_out*W_out)+Col]=Pvalue;

      }
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
    if (C < 3){
        float* k = (float*)malloc(sizeof(float)* M*C*K*K);
        cudaMemcpy(k, w.dptr_, sizeof(float)* M*C*K*K, cudaMemcpyDeviceToHost);
        cudaMemcpyToSymbol(k_const,k, sizeof(float)*C*K*K*M,0);

        dim3 gridDim(ceil(W*1.0/TILE_WIDTH_SMALL),ceil(H*1.0/TILE_WIDTH_SMALL),B);
        dim3 blockDim(TILE_WIDTH_SMALL,TILE_WIDTH_SMALL,1);
        small_kernel<<<gridDim,blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K,H_out,W_out);
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
        return;
    }

    float* k = (float*)malloc(sizeof(float)* M*C*K*K);
    cudaMemcpy(k, w.dptr_, sizeof(float)* M*C*K*K, cudaMemcpyDeviceToHost);
    cudaMemcpyToSymbol(k_const,k, sizeof(float)*C*K*K*M,0);

    dim3 DimGridMM(ceil(H_out*W_out*1.0/TILE_WIDTH),ceil(M*1.0/TILE_WIDTH), B);
    dim3 DimBlockMM(TILE_WIDTH,TILE_WIDTH,1);
    
    forward_kernel<<<DimGridMM,DimBlockMM>>>(C, K, H, W, M,x.dptr_,y.dptr_, H_out, W_out);


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