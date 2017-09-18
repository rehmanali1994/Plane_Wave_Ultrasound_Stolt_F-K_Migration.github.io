// To Compile: nvcc fkmigCUDA.cu -o fkmigCUDA.out -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcufft
// To Run: ./fkmigCUDA.out SIG.txt nt nx fs pitch TXangle c t0 migSIG.txt

#include <stdio.h>
#include <cuda.h>
#include <typeinfo>
#include <iostream>
#include <mex.h>
#include <helper_cuda.h>
#include <cufft.h>
#include <math.h>

// Define macros for excessively long names
#define pi 3.141592653589793238462643383279502884197169399375105820974f
#define CCE checkCudaErrors
#define HtoD cudaMemcpyHostToDevice
#define DtoH cudaMemcpyDeviceToHost

// CUDA Texture Objects for Real and Imaginary Parts of Spatiotemporal Frequency Domain of Signals
static cudaTextureObject_t texObjReal;
static cudaTextureDesc texDescReal;
static cudaResourceDesc resDescReal;
static cudaTextureObject_t texObjImag;
static cudaTextureDesc texDescImag;
static cudaResourceDesc resDescImag;

// Runs batched FFT and IFFT on device data
void batchedFFT(cufftComplex* dData, int N, int BATCH) {
	cufftHandle plan;
	if (cufftPlan1d(&plan, N, CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
	}
	if (cufftExecC2C(plan, dData, dData, CUFFT_FORWARD) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
	}
	if (cudaThreadSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	}
}
void batchedIFFT(cufftComplex* dData, int N, int BATCH) {
	cufftHandle plan;
	if (cufftPlan1d(&plan, N, CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
	}
	if (cufftExecC2C(plan, dData, dData, CUFFT_INVERSE) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
	}
	if (cudaThreadSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	}
}

// Outputs Matrix Transpose
__global__ void transpose(cufftComplex *odata, cufftComplex *idata, int numRows, int numCols)
{
	int c_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int r_idx = blockIdx.y*blockDim.y + threadIdx.y;

	if (r_idx < numRows && c_idx < numCols) {
		float origx = idata[c_idx + numCols * r_idx].x;
		float origy = idata[c_idx + numCols * r_idx].y;
		__syncthreads();
		odata[r_idx + numRows * c_idx].x = origx;
		odata[r_idx + numRows * c_idx].y = origy;
	}
}

// Trim the RF Signals
__global__ void rfTrim(cufftComplex *SIG, int nf, int nx, int nxFFT, float *dt, float *f0, float t0)
{
	int x_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int f_idx = blockIdx.y*blockDim.y + threadIdx.y;

	if (x_idx < nx && f_idx < nf) {
		float realSIG = SIG[x_idx + f_idx * nxFFT].x;
		float imagSIG = SIG[x_idx + f_idx * nxFFT].y;
		float phase = -2 * pi*(dt[x_idx] + t0)*f0[f_idx];
		SIG[x_idx + f_idx * nxFFT].x = realSIG * cosf(phase) - imagSIG * sinf(phase);
		SIG[x_idx + f_idx * nxFFT].y = realSIG * sinf(phase) + imagSIG * cosf(phase);
	}
}

// Remove Evanescent Parts in Spatio-temporal Frequency Domain of the Signals
__global__ void removeEvanescent(cufftComplex *SIG, float *f0, int nf, float *kx, int nxFFT, float c)
{
	int kx_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int f0_idx = blockIdx.y*blockDim.y + threadIdx.y;

	if (kx_idx < nxFFT && f0_idx < nf) {
		if (abs(f0[f0_idx]) * abs(kx[kx_idx]) < c) {
			SIG[kx_idx + f0_idx * nxFFT].x = 0;
			SIG[kx_idx + f0_idx * nxFFT].y = 0;
		}
	}
}

// Get Real and Imaginary Parts
__global__ void getRealAndImag(cufftComplex *cmpdata, float *realdata, float *imagdata, int numRows, int numCols)
{
	int c_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int r_idx = blockIdx.y*blockDim.y + threadIdx.y;
	if (r_idx < numRows && c_idx < numCols) {
		realdata[c_idx + numCols * r_idx] = cmpdata[c_idx + numCols * r_idx].x;
		imagdata[c_idx + numCols * r_idx] = cmpdata[c_idx + numCols * r_idx].y;
	}
}

// Run Stolt Mapping Kernel
__global__ void stoltmap(cufftComplex *SIG, float *f0, float *kx, int ntFFT, int nxFFT, float c, float v, float beta, float fs, cudaTextureObject_t texObjReal, cudaTextureObject_t texObjImag)
{
	int f0_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int kx_idx = blockIdx.y*blockDim.y + threadIdx.y;

	if (kx_idx == 0 && f0_idx == 0) {
		SIG[0].x = 0;
		SIG[0].y = 0;
	}
	else if (kx_idx < nxFFT && f0_idx < ntFFT / 2 + 1) {
		// Note: we choose kz = 2*f/c (i.e. z = c*t/2)
		float Kx = kx[kx_idx];
		float f = f0[f0_idx];
		float fkz = v*sqrt(Kx*Kx + 4 * ((f*f) / (c*c)) / (beta*beta));
		__syncthreads();
		// Linear interpolation in the frequency domain: f -> fkz
		float fkz_idx = (fkz / (fs / ntFFT)) + 0.5f;
		float SIGreal = tex1DLayered<float>(texObjReal, fkz_idx, kx_idx);
		float SIGimag = tex1DLayered<float>(texObjImag, fkz_idx, kx_idx);
		__syncthreads();
		// Multiply By Obliquity factor: f / fkz
		SIG[kx_idx + f0_idx * nxFFT].x = SIGreal * f / fkz;
		SIG[kx_idx + f0_idx * nxFFT].y = SIGimag * f / fkz;
		__syncthreads();
	}
}

// Concatenate Negative Axial Frequencies to Fourier Domain of Migrated Solution
__global__ void concatNegAxialFreq(cufftComplex *SIG, cufftComplex *SIGfromTexture, int ntFFT, int nxFFT)
{
	int kx_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int f0_idx = blockIdx.y*blockDim.y + threadIdx.y;

	if (kx_idx < nxFFT && f0_idx < ntFFT) {
		if (f0_idx < ntFFT / 2 + 1) {
			// Original Part
			SIG[kx_idx + f0_idx * nxFFT].x = SIGfromTexture[kx_idx + f0_idx * nxFFT].x;
			SIG[kx_idx + f0_idx * nxFFT].y = SIGfromTexture[kx_idx + f0_idx * nxFFT].y;
		}
		else {
			// Concatenated Part
			SIG[kx_idx + f0_idx * nxFFT].x = SIGfromTexture[((nxFFT - kx_idx) % nxFFT) + (ntFFT - f0_idx) * nxFFT].x;
			SIG[kx_idx + f0_idx * nxFFT].y = -SIGfromTexture[((nxFFT - kx_idx) % nxFFT) + (ntFFT - f0_idx) * nxFFT].y;
		}
	}
}

// Steering Angle Compensation for RF Signals
__global__ void steerComp(cufftComplex *SIG, int nxFFT, int ntFFT, float *kx, float fs, float c, float gamma)
{
	int kx_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int t_idx = blockIdx.y*blockDim.y + threadIdx.y;

	if (kx_idx < nxFFT && t_idx < ntFFT) {
		float realSIG = SIG[kx_idx + t_idx * nxFFT].x;
		float imagSIG = SIG[kx_idx + t_idx * nxFFT].y;
		float dx = -gamma*t_idx / fs * c / 2;
		float phase = -2 * pi * kx[kx_idx] * dx;
		SIG[kx_idx + t_idx * nxFFT].x = realSIG * cosf(phase) - imagSIG * sinf(phase);
		SIG[kx_idx + t_idx * nxFFT].y = realSIG * sinf(phase) + imagSIG * cosf(phase);
	}
}


// Gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	// Argument check
	if (nrhs != 6)	{ mexErrMsgTxt("Wrong number of inputs.\n"); }
  if (nlhs != 1)	{ mexErrMsgTxt("Wrong number of outputs.\n"); }

  // Gather values from inputs
  double *SIGinput = (double *)mxGetData(prhs[0]);
  const mwSize *dimsSIG = mxGetDimensions(prhs[0]);
  mwSize nx = dimsSIG[0]; // Number of Array Elements [ROWS]
  mwSize nt = dimsSIG[1]; // Number of Time Points [COLUMNS]
  float fs = mxGetScalar(prhs[1]); // Frequency [Hz]
  float pitch = mxGetScalar(prhs[2]); // Element Pitch [m]
	float TXangle = mxGetScalar(prhs[3]); // TX Angle [rad]
	float c = mxGetScalar(prhs[4]); // Sound Speed [m/s]
	float t0 = mxGetScalar(prhs[5]); // Acquisition Start Time [s]

  // Zero-padding before FFTs
	// Time direction: extensive zero-padding is required with linear interpolation
	int ntshift = (int)(2 * ceil(t0*fs / 2));
	int ntFFT = 4 * nt + ntshift;
	// X direction: in order to avoid lateral edge effects
	float factor = 1.5f;
	int nxFFT = (int)(2 * ceil(factor*nx / 2));
	// Write values in for f0
	float* f0 = (float *)malloc(sizeof(float) * (ntFFT / 2 + 1));
	for (int i = 0; i < ntFFT / 2 + 1; i++)
		f0[i] = (float)i*fs / ntFFT;
	// Write values in for kx
	float* kx = (float *)malloc(sizeof(float) * nxFFT);
	for (int i = 0; i < nxFFT; i++)
		kx[i] = (float)((i > nxFFT / 2) ? i - nxFFT : i) / pitch / nxFFT;
	// Convert both f0 and kx to device arrays
	float *d_f0, *d_kx;
	CCE(cudaMalloc(&d_f0, (ntFFT / 2 + 1) * sizeof(float)));
	CCE(cudaMalloc(&d_kx, nxFFT * sizeof(float)));
	CCE(cudaMemcpy(d_f0, f0, (ntFFT / 2 + 1) * sizeof(float), HtoD));
	CCE(cudaMemcpy(d_kx, kx, nxFFT * sizeof(float), HtoD));

	// Read Signals Into Host Array and Copy to Device
	cufftComplex *SIG = (cufftComplex *)malloc(ntFFT*nxFFT*sizeof(cufftComplex));
	for (int jj = 0; jj < ntFFT; jj++) {
		for (int ii = 0; ii < nxFFT; ii++) {
			SIG[ii + jj*nxFFT].x = 0;
			SIG[ii + jj*nxFFT].y = 0;
		}
	}
	for (int jj = 0; jj < nt; jj++) {
		for (int ii = 0; ii < nx; ii++) {
			SIG[ii + jj*nxFFT].x = (float) SIGinput[ii + jj*nx];
			SIG[ii + jj*nxFFT].y = 0;
		}
	}
	cufftComplex *d_SIG, *d_SIG_t;
	CCE(cudaMalloc(&d_SIG, ntFFT * nxFFT * sizeof(cufftComplex)));
	CCE(cudaMalloc(&d_SIG_t, ntFFT * nxFFT * sizeof(cufftComplex)));
	CCE(cudaMemcpy(d_SIG, SIG, ntFFT * nxFFT * sizeof(cufftComplex), HtoD));

	// Take Temporal FFT
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid((nxFFT + dimBlock.x - 1) / dimBlock.x,
		(ntFFT + dimBlock.y - 1) / dimBlock.y, 1);
	transpose << <dimGrid, dimBlock >> >(d_SIG_t, d_SIG, ntFFT, nxFFT);
	batchedFFT(d_SIG_t, ntFFT, nxFFT);
	dim3 dimBlockT(16, 16, 1);
	dim3 dimGridT((ntFFT + dimBlock.x - 1) / dimBlock.x,
		(nxFFT + dimBlock.y - 1) / dimBlock.y, 1);
	transpose << <dimGridT, dimBlockT >> >(d_SIG, d_SIG_t, nxFFT, ntFFT);

	// ERM velocity
	float sinA = sinf(TXangle);
	float cosA = cosf(TXangle);
	float v = c / sqrt(1 + cosA + sinA * sinA);

	// Compensate for steering angle and/or depth start
	// Assumes that t=0 is when any element first sends its tx
	float* dt = (float *)malloc(sizeof(float) * nx);
	for (int i = 0; i < nx; i++)
		dt[i] = (float)((TXangle < 0) ? nx - 1 - i : -i)*sinA*pitch / c;
	float *d_dt;
	CCE(cudaMalloc(&d_dt, nx * sizeof(float)));
	CCE(cudaMemcpy(d_dt, dt, nx * sizeof(float), cudaMemcpyHostToDevice));
	rfTrim << <dimGrid, dimBlock >> >(d_SIG, ntFFT / 2 + 1, nx, nxFFT, d_dt, d_f0, t0);

	// Take Azimuthal (Spatial) FFT
	batchedFFT(d_SIG, nxFFT, ntFFT / 2 + 1);

	// Perform Stolt Mapping
	removeEvanescent << <dimGrid, dimBlock >> >(d_SIG, d_f0, ntFFT / 2 + 1, d_kx, nxFFT, c);
	// Separate real and imaginary components
	cufftComplex *d_SIGforTexture;
	CCE(cudaMalloc(&d_SIGforTexture, (ntFFT / 2 + 1) * nxFFT * sizeof(cufftComplex)));
	transpose << <dimGrid, dimBlock >> >(d_SIGforTexture, d_SIG, ntFFT / 2 + 1, nxFFT);
	float *d_SIGreal, *d_SIGimag;
	CCE(cudaMalloc(&d_SIGreal, (ntFFT / 2 + 1) * nxFFT * sizeof(float)));
	CCE(cudaMalloc(&d_SIGimag, (ntFFT / 2 + 1) * nxFFT * sizeof(float)));
	dim3 dimBlockTex(16, 16, 1);
	dim3 dimGridTex((ntFFT / 2 + dimBlock.x) / dimBlock.x,
		(nxFFT + dimBlock.y - 1) / dimBlock.y, 1);
	getRealAndImag << <dimGridTex, dimBlockTex >> >(d_SIGforTexture, d_SIGreal, d_SIGimag, nxFFT, ntFFT / 2 + 1);
	// Write real and imaginary parts back to host memory
	float *SIGreal = (float *)malloc(nxFFT * (ntFFT / 2 + 1) * sizeof(float));
	float *SIGimag = (float *)malloc(nxFFT * (ntFFT / 2 + 1) * sizeof(float));
	CCE(cudaMemcpy(SIGreal, d_SIGreal, nxFFT * (ntFFT / 2 + 1) * sizeof(float), DtoH));
	CCE(cudaMemcpy(SIGimag, d_SIGimag, nxFFT * (ntFFT / 2 + 1) * sizeof(float), DtoH));

	// Make the Spatio-Temporal Fourier Domain of the Signals a Texture
	// Real Part
	// Create CUDA Array to Interpolate on
	cudaExtent extentDescReal = make_cudaExtent(ntFFT/2+1, 0, nxFFT);  // <-- 0 height required for 1D-Layered
	cudaChannelFormatDesc channelDescReal = cudaCreateChannelDesc<float>();
	cudaMemcpy3DParms mParamsReal = { 0 };
	mParamsReal.srcPtr = make_cudaPitchedPtr(SIGreal, (ntFFT / 2 + 1) * sizeof(float), ntFFT / 2 + 1, 1);
	mParamsReal.kind = cudaMemcpyHostToDevice;
	mParamsReal.extent = make_cudaExtent(ntFFT / 2 + 1, 1, nxFFT);
	cudaArray* cuArrayReal;
	CCE(cudaMalloc3DArray(&cuArrayReal, &channelDescReal, extentDescReal, cudaArrayLayered));
	mParamsReal.dstArray = cuArrayReal;
	CCE(cudaMemcpy3D(&mParamsReal));
	// Texture Description
	texDescReal.addressMode[0] = cudaAddressModeBorder;
	texDescReal.filterMode = cudaFilterModeLinear;
	texDescReal.normalizedCoords = 0; // false -- Not using normalized coords
	texDescReal.readMode = cudaReadModeElementType;
	// Resource Description
	resDescReal.resType = cudaResourceTypeArray;
  resDescReal.res.array.array = cuArrayReal;
	// Create CUDA Texture Object from Texture and Resource Descriptions
	CCE(cudaCreateTextureObject(&texObjReal, &resDescReal, &texDescReal, NULL));
	// Imaginary Part
	// Create CUDA Array to Interpolate on
	cudaExtent extentDescImag = make_cudaExtent(ntFFT / 2 + 1, 0, nxFFT);  // <-- 0 height required for 1D-Layered
	cudaChannelFormatDesc channelDescImag = cudaCreateChannelDesc<float>();
	cudaMemcpy3DParms mParamsImag = { 0 };
	mParamsImag.srcPtr = make_cudaPitchedPtr(SIGimag, (ntFFT / 2 + 1) * sizeof(float), ntFFT / 2 + 1, 1);
	mParamsImag.kind = cudaMemcpyHostToDevice;
	mParamsImag.extent = make_cudaExtent(ntFFT / 2 + 1, 1, nxFFT);
	cudaArray* cuArrayImag;
	CCE(cudaMalloc3DArray(&cuArrayImag, &channelDescImag, extentDescImag, cudaArrayLayered));
	mParamsImag.dstArray = cuArrayImag;
	CCE(cudaMemcpy3D(&mParamsImag));
	// Texture Description
	texDescImag.addressMode[0] = cudaAddressModeBorder;
	texDescImag.filterMode = cudaFilterModeLinear;
	texDescReal.normalizedCoords = 0; // false -- Not using normalized coords
	texDescImag.readMode = cudaReadModeElementType;
	// Resource Description
	resDescImag.resType = cudaResourceTypeArray;
  resDescImag.res.array.array = cuArrayImag;
	// Create CUDA Texture Object from Texture and Resource Descriptions
	CCE(cudaCreateTextureObject(&texObjImag, &resDescImag, &texDescImag, NULL));

	// Invoke Stolt Mapping Kernel
	float beta = (1 + cosA) * sqrt(1 + cosA) / (1 + cosA + sinA * sinA);
	stoltmap << <dimGridTex, dimBlockTex >> >(d_SIGforTexture, d_f0, d_kx, ntFFT, nxFFT, c, v, beta, fs, texObjReal, texObjImag);

	// Take Axial IFFT
	concatNegAxialFreq << <dimGrid, dimBlock >> >(d_SIG, d_SIGforTexture, ntFFT, nxFFT);
	transpose << <dimGrid, dimBlock >> >(d_SIG_t, d_SIG, ntFFT, nxFFT);
	batchedIFFT(d_SIG_t, ntFFT, nxFFT);
	transpose << <dimGridT, dimBlockT >> >(d_SIG, d_SIG_t, nxFFT, ntFFT);

	// Steering Angle Compensation
	float gamma = sinA / (2 - cosA);
	steerComp << <dimGrid, dimBlock >> >(d_SIG, nxFFT, ntFFT, d_kx, fs, c, gamma);

	// Take Spatial IFFT
	batchedIFFT(d_SIG, nxFFT, ntFFT);
	CCE(cudaMemcpy(SIG, d_SIG, ntFFT * nxFFT * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

	// Write final migrated signal to file
	plhs[0] = mxCreateDoubleMatrix( nx, nt, mxREAL);
  double *migSIG = (double *)mxGetPr(plhs[0]);
	for (int jj = 0; jj < nt; jj++) {
		for (int ii = 0; ii < nx; ii++) {
			migSIG[ii + jj*nx] = (double) SIG[ii + (jj+ntshift)*nxFFT].x;
		}
	}

	// Destroy all allocations and reset all state on the current device in the current process.
	CCE(cudaDeviceReset());

}
