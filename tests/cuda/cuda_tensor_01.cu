// ---------------------------------------------------------------------
//
// Copyright (C) 2016 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// Test operator[] and norm_square of cuda_tensor.

#include <deal.II/base/logstream.h>
#include <deal.II/base/tensor.h>

#include <fstream>
#include <iomanip>

#include "../tests.h"

void
test_cpu()
{
  double             a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  const unsigned int dim     = 3;
  Tensor<2, dim>     t;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      t[i][j] = a[i][j];

  Tensor<2, dim> t_copy;
  std::memcpy(&t_copy, &t, 9);
  deallog.push("values");
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      deallog << t_copy[i][j] << std::endl;
  deallog.pop();

  Vector<double> unrolled(9);
  t_copy.unroll(unrolled);
  for (unsigned int i = 0; i < unrolled.size(); ++i)
    std::cout << unrolled(i) << std::endl;

  deallog.push("norm_square");
  deallog << t.norm_square() << std::endl;
  deallog.pop();
}

__global__ void init_kernel(Tensor<2, 3> *t, const unsigned int N)
{
  const unsigned int i = threadIdx.y;
  const unsigned int j = threadIdx.x;
  if ((i < N) && (j < N))
    {
      (*t)[i][j] = j + i * N + 1.;
      printf("%f\n", (*t)[i][j]);
    }
}

template <int rank, int dim>
__global__ void
unroll_kernel(Tensor<rank, dim> *t, double *unrolled)
{
  const unsigned int     i    = threadIdx.x;
  constexpr unsigned int size = Utilities::pow(rank, dim);
  if (i < size)
    unrolled[i] = t[t->unrolled_to_component_indices(i)];
}

__global__ void norm_kernel(Tensor<2, 3> *t, double *norm)
{
  if (threadIdx.x == 0)
    *norm = t->norm_square();
}

void
test_gpu()
{
  const unsigned int dim = 3;
  double *           norm_dev;
  double             norm_host;
  Tensor<2, dim> *   t_dev;

  // Allocate objects on the device
  cudaError_t cuda_error = cudaMalloc(&t_dev, sizeof(Tensor<2, dim>));
  AssertCuda(cuda_error);
  cuda_error = cudaMalloc(&norm_dev, sizeof(double));
  AssertCuda(cuda_error);

  // Launch the kernels.
  dim3 block_dim(dim, dim);
  init_kernel<<<1, block_dim>>>(t_dev, dim);
  norm_kernel<<<1, 1>>>(t_dev, norm_dev);

  // Copy the result to the host
  cuda_error =
    cudaMemcpy(&norm_host, norm_dev, sizeof(double), cudaMemcpyDeviceToHost);
  AssertCuda(cuda_error);

  double *unrolled_dev;
  cuda_error = cudaMalloc(unrolled_dev, dim * dim * sizeof(double));
  AssertCuda(cuda_error);
  unroll_kernel<<<1, dim * dim>>>(&t_dev, unrolled_dev);
  std::vector<double> unrolled(dim * dim);
  cuda_error = cudaMemcpy(&t, t_dev, sizeof(double), cudaMemcpyDeviceToHost);
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      {
        deallog << t[i][j] << std::endl;
      }


  // Free memory
  cuda_error = cudaFree(t_dev);
  AssertCuda(cuda_error);
  cuda_error = cudaFree(norm_dev);
  AssertCuda(cuda_error);

  // Output result
  deallog.push("norm_square GPU");
  deallog << norm_host << std::endl;
}

int
main()
{
  std::ofstream logfile("output");
  deallog << std::setprecision(5);
  deallog.attach(logfile);

  init_cuda();

  test_cpu();

  test_gpu();
}
