__global__ void unrollk_kernel(int M, int C, int K, const float* k, float* unroll_k){
    /*
     * DimGrid(ceil(M/TILE_WIDTH),ceil(C*K*K*1.0/TILE_WIDTH),1)
     * DimBlock(TILE_WIDTH, TILE_WIDTH,1)
     * thread mapped onto unroll_k array
     * unroll_k shape: B*M*(C*K*K)
     * observation:  for all batches k_unroll is the same so should load it into constant memory. 
     * !! unroll_k is of size M*C*K*K <3000 for given data sets, so i suggest use cpu to do unrolling
     */
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define unroll_k2d(y,x) unroll_k[(y)*(C*K*K)+(x)]
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int posx = bx*TILE_WIDTH+tx;
    int m = by*TILE_WIDTH+ty; // (posx, posy=m) = locattion in unroll_k

    int c = posx/(K*K);
    int filterx= (posx%(K*K))%K;
    int filtery = (posx%(K*K))/K; // (filterx, filtery) = location in k*K filter

    if(posx<C*K*K && m<M){
        unroll_k3d(m,posx)=k4d(m,c,filtery, filterx);
    }
    #undef k4d
    #undef unroll_k3d
}
