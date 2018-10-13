// ---------------------------------------------------------------------
//
// Copyright (C) 2018 by the deal.II authors
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


// Check Utilities::MPI::sum for an array living in CUDA memory space.

#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/utilities.h>

#include "../tests.h"

void
test()
{
  constexpr unsigned int n = 10;

  std::unique_ptr<double[], void (*)(double *)> dev_array_value_ptr(
    nullptr, Utilities::CUDA::delete_device_data<double>);
  dev_array_value_ptr.reset(Utilities::CUDA::allocate_device_data<double>(n));

  const unsigned int my_id = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  std::vector<double> cpu_array(n);
  for (unsigned int i = 0; i < n; ++i)
    cpu_array[i] = i + my_id;

  Utilities::CUDA::copy_to_dev(cpu_array, dev_array_value_ptr.get());

  std::unique_ptr<double[], void (*)(double *)> dev_array_output_ptr(
    nullptr, Utilities::CUDA::delete_device_data<double>);
  dev_array_output_ptr.reset(Utilities::CUDA::allocate_device_data<double>(n));

  const ArrayView<const double, MemorySpace::CUDA> value_array_view(
    dev_array_value_ptr.get(), n);
  const ArrayView<double, MemorySpace::CUDA> output_array_view(
    dev_array_output_ptr.get(), n);

  Utilities::MPI::sum(value_array_view, MPI_COMM_WORLD, output_array_view);
  Utilities::CUDA::copy_to_host(dev_array_output_ptr.get(), cpu_array);

  for (double val : cpu_array)
    deallog << val << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  mpi_initlog();

  unsigned int myid = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  deallog.push(Utilities::int_to_string(myid));

  // By default, all the ranks will try to access the device 0. This is fine if
  // we have one rank per node _and_ one gpu per node. If we have multiple GPUs
  // on one node, we need each process to access a different GPU. We assume that
  // each node has the same number of GPUs.
  int         n_devices       = 0;
  cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
  AssertCuda(cuda_error_code);
  int device_id   = myid % n_devices;
  cuda_error_code = cudaSetDevice(device_id);
  AssertCuda(cuda_error_code);

  test();

  deallog << "OK" << std::endl;

  return 0;
}
