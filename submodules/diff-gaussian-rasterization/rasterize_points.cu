/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <vector>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include "cuda_rasterizer/auxiliary.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);
  float* out_invdepthptr = nullptr;

  out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  out_invdepthptr = out_invdepth.data<float>();

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_invdepthptr,
		antialiasing,
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_invdepth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());
  
  float* dL_dinvdepthsptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  if(dL_dout_invdepth.size(0) != 0)
  {
	dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
	dL_dinvdepths = dL_dinvdepths.contiguous();
	dL_dinvdepthsptr = dL_dinvdepths.data<float>();
	dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
  }

  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_invdepthptr,
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dinvdepthsptr,
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  antialiasing,
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

__global__ void tileResidualKernel(
	const float* residual,
	int H, int W,
	float* out_tile,
	int tiles_x,
	int tiles_y)
{
	int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
	int total = H * W;
	if (idx >= total) return;

	int y = idx / W;
	int x = idx - y * W;
	int tile_x = x / BLOCK_X;
	int tile_y = y / BLOCK_Y;
	if (tile_x >= tiles_x || tile_y >= tiles_y) return;

	int tile_id = tile_y * tiles_x + tile_x;
	int plane = H * W;
	atomicAdd(out_tile + tile_id * 3 + 0, residual[idx]);
	atomicAdd(out_tile + tile_id * 3 + 1, residual[plane + idx]);
	atomicAdd(out_tile + tile_id * 3 + 2, residual[plane * 2 + idx]);
}

__global__ void tileMomentsKernel(
	const float* residual,
	int H, int W,
	float* out_tile,
	float* out_energy,
	int tiles_x,
	int tiles_y)
{
	int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
	int total = H * W;
	if (idx >= total) return;

	int y = idx / W;
	int x = idx - y * W;
	int tile_x = x / BLOCK_X;
	int tile_y = y / BLOCK_Y;
	if (tile_x >= tiles_x || tile_y >= tiles_y) return;

	int tile_id = tile_y * tiles_x + tile_x;
	int plane = H * W;
	float r0 = residual[idx];
	float r1 = residual[plane + idx];
	float r2 = residual[plane * 2 + idx];
	atomicAdd(out_tile + tile_id * 3 + 0, r0);
	atomicAdd(out_tile + tile_id * 3 + 1, r1);
	atomicAdd(out_tile + tile_id * 3 + 2, r2);
	atomicAdd(out_energy + tile_id, r0 * r0 + r1 * r1 + r2 * r2);
}

torch::Tensor computeTileResidualCUDA(
	const torch::Tensor& residual)
{
	TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
	TORCH_CHECK(residual.dim() == 3 && residual.size(0) == 3, "residual must have shape [3, H, W]");
	TORCH_CHECK(residual.scalar_type() == at::kFloat, "residual must be float32");

	auto res = residual.contiguous();
	const int H = res.size(1);
	const int W = res.size(2);
	const int tiles_x = (W + BLOCK_X - 1) / BLOCK_X;
	const int tiles_y = (H + BLOCK_Y - 1) / BLOCK_Y;
	const int num_tiles = tiles_x * tiles_y;

	auto out = torch::zeros({num_tiles, 3}, res.options().dtype(torch::kFloat32));

	const int threads = 256;
	const int total = H * W;
	const int blocks = (total + threads - 1) / threads;
	tileResidualKernel<<<blocks, threads>>>(
		res.data_ptr<float>(),
		H, W,
		out.data_ptr<float>(),
		tiles_x, tiles_y);
	auto kernel_err = cudaGetLastError();
	TORCH_CHECK(kernel_err == cudaSuccess, "tileResidualKernel failed: ", cudaGetErrorString(kernel_err));

	return out;
}

std::vector<torch::Tensor> computeTileMomentsCUDA(
	const torch::Tensor& residual)
{
	TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
	TORCH_CHECK(residual.dim() == 3 && residual.size(0) == 3, "residual must have shape [3, H, W]");
	TORCH_CHECK(residual.scalar_type() == at::kFloat, "residual must be float32");

	auto res = residual.contiguous();
	const int H = res.size(1);
	const int W = res.size(2);
	const int tiles_x = (W + BLOCK_X - 1) / BLOCK_X;
	const int tiles_y = (H + BLOCK_Y - 1) / BLOCK_Y;
	const int num_tiles = tiles_x * tiles_y;

	auto out = torch::zeros({num_tiles, 3}, res.options().dtype(torch::kFloat32));
	auto out_energy = torch::zeros({num_tiles}, res.options().dtype(torch::kFloat32));

	const int threads = 256;
	const int total = H * W;
	const int blocks = (total + threads - 1) / threads;
	tileMomentsKernel<<<blocks, threads>>>(
		res.data_ptr<float>(),
		H, W,
		out.data_ptr<float>(),
		out_energy.data_ptr<float>(),
		tiles_x, tiles_y);
	auto kernel_err = cudaGetLastError();
	TORCH_CHECK(kernel_err == cudaSuccess, "tileMomentsKernel failed: ", cudaGetErrorString(kernel_err));

	return {out, out_energy};
}

__global__ void computeFwScoreKernel(
	int P,
	const uint32_t* tiles_touched,
	const uint32_t* point_offsets,
	const uint64_t* point_list_keys_unsorted,
	const float* tile_residual,
	const float* tile_energy,
	int num_tiles,
	int tiles_x,
	const float2* means2D,
	const float4* conic_opacity,
	const int* radii,
	float* out_score,
	int score_mode)
{
	int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
	if (idx >= P) return;

	if (radii[idx] <= 0)
	{
		out_score[idx] = 0.0f;
		return;
	}

	uint32_t count = tiles_touched[idx];
	if (count == 0)
	{
		out_score[idx] = 0.0f;
		return;
	}

	uint32_t off = (idx == 0) ? 0 : point_offsets[idx - 1];

	float s0 = 0.0f;
	float s1 = 0.0f;
	float s2 = 0.0f;
	float sum_w = 0.0f;
	float sum_e = 0.0f;
	const bool use_weight = (score_mode >= 3);
	const bool use_energy = (score_mode == 2 || score_mode == 4);
	const float2 mean = means2D[idx];
	const float4 conic = conic_opacity[idx];
	for (uint32_t j = 0; j < count; ++j)
	{
		uint64_t key = point_list_keys_unsorted[off + j];
		uint32_t tile = (uint32_t)(key >> 32);
		if (tile < (uint32_t)num_tiles)
		{
			float w = 1.0f;
			if (use_weight)
			{
				const int tile_y = (int)(tile / (uint32_t)tiles_x);
				const int tile_x = (int)(tile - (uint32_t)tile_y * (uint32_t)tiles_x);
				const float cx = tile_x * BLOCK_X + 0.5f * (BLOCK_X - 1);
				const float cy = tile_y * BLOCK_Y + 0.5f * (BLOCK_Y - 1);
				const float dx = cx - mean.x;
				const float dy = cy - mean.y;
				const float qf = conic.x * dx * dx + 2.0f * conic.y * dx * dy + conic.z * dy * dy;
				const float power = -0.5f * qf;
				if (power < -50.0f)
					w = 0.0f;
				else
					w = expf(power);
			}
			const float* tr = tile_residual + tile * 3;
			s0 += w * tr[0];
			s1 += w * tr[1];
			s2 += w * tr[2];
			sum_w += w;
			if (use_energy)
			{
				sum_e += w * tile_energy[tile];
			}
		}
	}

	if (score_mode == 0)
	{
		out_score[idx] = sqrtf(s0 * s0 + s1 * s1 + s2 * s2);
		return;
	}

	if (sum_w <= 0.0f)
	{
		out_score[idx] = 0.0f;
		return;
	}

	if (score_mode == 1 || score_mode == 3)
	{
		const float inv = 1.0f / sum_w;
		s0 *= inv; s1 *= inv; s2 *= inv;
		out_score[idx] = sqrtf(s0 * s0 + s1 * s1 + s2 * s2);
		return;
	}

	// score_mode == 2 or 4: RMS of residual magnitude
	const float inv = 1.0f / sum_w;
	const float mean_e = sum_e * inv;
	out_score[idx] = sqrtf(fmaxf(mean_e, 0.0f));
}

torch::Tensor computeFwScoreCUDA(
	const torch::Tensor& tile_residual,
	const torch::Tensor& tile_energy,
	const int tiles_x,
	const torch::Tensor& radii,
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const int score_mode)
{
	TORCH_CHECK(tile_residual.is_cuda(), "tile_residual must be a CUDA tensor");
	TORCH_CHECK(tile_energy.is_cuda(), "tile_energy must be a CUDA tensor");
	TORCH_CHECK(radii.is_cuda(), "radii must be a CUDA tensor");
	TORCH_CHECK(geomBuffer.is_cuda(), "geomBuffer must be a CUDA tensor");
	TORCH_CHECK(binningBuffer.is_cuda(), "binningBuffer must be a CUDA tensor");
	TORCH_CHECK(tile_residual.dim() == 2 && tile_residual.size(1) == 3, "tile_residual must have shape [num_tiles, 3]");
	TORCH_CHECK(tile_energy.dim() == 1, "tile_energy must have shape [num_tiles]");

	const int P = radii.size(0);
	auto out_score = torch::zeros({P}, tile_residual.options().dtype(torch::kFloat32));
	if (P == 0)
	{
		return out_score;
	}

	auto tile = tile_residual.contiguous();
	auto energy = tile_energy.contiguous();
	auto radii_c = radii.contiguous();

	char* geom_chunk = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
	CudaRasterizer::GeometryState geomState = CudaRasterizer::GeometryState::fromChunk(geom_chunk, P);

	uint32_t num_rendered = 0;
	auto copy_err = cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	TORCH_CHECK(copy_err == cudaSuccess, "cudaMemcpy failed: ", cudaGetErrorString(copy_err));

	char* binning_chunk = reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr());
	CudaRasterizer::BinningState binningState = CudaRasterizer::BinningState::fromChunk(binning_chunk, num_rendered);

	const int num_tiles = (int)tile.size(0);
	TORCH_CHECK(num_tiles == (int)energy.size(0), "tile_residual and tile_energy must have same num_tiles");
	TORCH_CHECK(tiles_x > 0, "tiles_x must be > 0");
	const int threads = 256;
	const int blocks = (P + threads - 1) / threads;
	computeFwScoreKernel<<<blocks, threads>>>(
		P,
		geomState.tiles_touched,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		tile.data_ptr<float>(),
		energy.data_ptr<float>(),
		num_tiles,
		tiles_x,
		geomState.means2D,
		geomState.conic_opacity,
		radii_c.data_ptr<int>(),
		out_score.data_ptr<float>(),
		score_mode);
	auto kernel_err = cudaGetLastError();
	TORCH_CHECK(kernel_err == cudaSuccess, "computeFwScoreKernel failed: ", cudaGetErrorString(kernel_err));

	return out_score;
}
