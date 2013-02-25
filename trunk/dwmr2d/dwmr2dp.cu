__global__ void ftcsKernel(float *Cxn, float *Cyn, float *Cxo, float *Cyo, float *diffu,float *diffd,float *diffl, float *diffr, float *T2val, float Adx, int dimX)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;// place in x dim
	int y = blockIdx.y; 						// place in y dim
	int ind = x+y*dimX;							// current index in linear space
	int yp1 = y+1;
	int ym1 = y-1;
	int xp1 = x+1;
	int xm1 = x-1;
	
	// periodic bc
	if (y==0) ym1 = gridDim.y-1;
	if (y==gridDim.y-1) yp1 = 0;
	if (x==0) xm1 = dimX-1;
	if (x==dimX-1) xp1 = 0;

	if (x >= 0 && x <= (dimX-1) && y >= 0 && y <= (gridDim.y-1) )
	{
		Cxn[ind] = Cxo[ind] - T2val[ind]*Cxo[ind]
			+ diffu[ind]*(cos(Adx)*Cxo[yp1*dimX+x] + sin(Adx)*Cyo[yp1*dimX+x] - Cxo[ind])
			+ diffd[ind]*(cos(Adx)*Cxo[ym1*dimX+x] - sin(Adx)*Cyo[ym1*dimX+x] - Cxo[ind])
			+ diffl[ind]*(Cxo[y*dimX+xp1] - Cxo[ind])
			+ diffr[ind]*(Cxo[y*dimX+xm1] - Cxo[ind]);
			
		Cyn[ind] = Cyo[ind] - T2val[ind]*Cyo[ind]
			+ diffu[ind]*(cos(Adx)*Cyo[yp1*dimX+x] - sin(Adx)*Cxo[yp1*dimX+x] - Cyo[ind])
			+ diffd[ind]*(cos(Adx)*Cyo[ym1*dimX+x] + sin(Adx)*Cxo[ym1*dimX+x] - Cyo[ind])
			+ diffl[ind]*(Cyo[y*dimX+xp1] - Cyo[ind])
			+ diffr[ind]*(Cyo[y*dimX+xm1] - Cyo[ind]);
	}
}
