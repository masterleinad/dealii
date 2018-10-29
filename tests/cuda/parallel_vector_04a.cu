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


// check that operator= resets ghosts, both if they have been set and if they
// have not been set

#include <deal.II/base/cuda.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <iostream>
#include <vector>

#include "../tests.h"

void
test0()
{
  const unsigned int my_owned_size = 2;
  const unsigned int my_ghost_size = 0;
  const unsigned int my_total_size = my_owned_size + my_ghost_size;

  double *const device_memory_pointer = [](const std::size_t size) {
    double *device_ptr;
    Utilities::CUDA::malloc(device_ptr, size);
    return device_ptr;
  }(my_total_size);
  std::vector<double> cpu_values(my_total_size);
  for (unsigned int i = 0; i < my_total_size; ++i)
    {
      cpu_values[i] = i + 10;
    }
  Utilities::CUDA::copy_to_dev(cpu_values, device_memory_pointer);

  double *const device_owned_pointer = device_memory_pointer;
  double *const device_ghost_pointer = device_memory_pointer + my_owned_size;

  const unsigned int n_import_targets = 1;
  const unsigned int n_ghost_targets  = 0;

  std::vector<MPI_Request> requests(n_import_targets + n_ghost_targets);

  if (n_ghost_targets > 0)
    {
      const int ierr = MPI_Irecv(device_ghost_pointer,
                                 1,
                                 MPI_DOUBLE,
                                 0, // source
                                 0, // channel
                                 MPI_COMM_WORLD,
                                 &requests[0]);
      AssertThrowMPI(ierr);
    }

  double *const device_temporary_memory_pointer = [](const std::size_t size) {
    double *device_ptr;
    Utilities::CUDA::malloc(device_ptr, size);
    return device_ptr;
  }(my_total_size);

  if (n_import_targets > 0)
    {
      const int ierr = MPI_Isend(device_owned_pointer,
                                 1,
                                 MPI_DOUBLE,
                                 1, // destination,
                                 0, // channel,
                                 MPI_COMM_WORLD,
                                 &requests[n_ghost_targets]);
      AssertThrowMPI(ierr);

      std::vector<double> cpu_values(1);
      Utilities::CUDA::copy_to_host(device_owned_pointer, cpu_values);
      for (const auto value : cpu_values)
        std::cout << value << std::endl;
    }
  for (auto request : requests)
    {
      MPI_Status status;
      MPI_Wait(&request, &status);
      int number_amount;
      MPI_Get_count(&status, MPI_DOUBLE, &number_amount);
      std::cout << "received " << number_amount << " elements from "
                << status.MPI_SOURCE << " with tag " << status.MPI_TAG
                << std::endl;
    }

  {
    std::vector<double> cpu_values(my_total_size);
    Utilities::CUDA::copy_to_host(device_memory_pointer, cpu_values);
    for (unsigned int j = 0; j < cpu_values.size(); ++j)
      std::cout << device_memory_pointer + j << " : " << cpu_values[j]
                << std::endl;
  }
}


void
test1()
{
  const unsigned int my_owned_size = 2;
  const unsigned int my_ghost_size = 1;
  const unsigned int my_total_size = my_owned_size + my_ghost_size;

  double *const device_memory_pointer = [](const std::size_t size) {
    double *device_ptr;
    Utilities::CUDA::malloc(device_ptr, size);
    return device_ptr;
  }(my_total_size);
  std::vector<double> cpu_values(my_total_size);
  for (unsigned int i = 0; i < my_total_size; ++i)
    {
      cpu_values[i] = i + 100;
    }
  Utilities::CUDA::copy_to_dev(cpu_values, device_memory_pointer);

  double *const device_owned_pointer = device_memory_pointer;
  double *const device_ghost_pointer = device_memory_pointer + my_owned_size;

  const unsigned int n_import_targets = 0;
  const unsigned int n_ghost_targets  = 1;

  std::vector<MPI_Request> requests(n_import_targets + n_ghost_targets);

  if (n_ghost_targets > 0)
    {
      const int ierr = MPI_Irecv(device_ghost_pointer,
                                 1,
                                 MPI_DOUBLE,
                                 0, // source
                                 0, // channel
                                 MPI_COMM_WORLD,
                                 &requests[0]);
      AssertThrowMPI(ierr);
    }

  double *const device_temporary_memory_pointer = [](const std::size_t size) {
    double *device_ptr;
    Utilities::CUDA::malloc(device_ptr, size);
    return device_ptr;
  }(my_total_size);

  if (n_import_targets > 0)
    {
      const int ierr = MPI_Isend(device_owned_pointer,
                                 1,
                                 MPI_DOUBLE,
                                 1, // destination,
                                 0, // channel,
                                 MPI_COMM_WORLD,
                                 &requests[0]);
      AssertThrowMPI(ierr);

      std::vector<double> cpu_values(1);
      Utilities::CUDA::copy_to_host(device_owned_pointer, cpu_values);
      for (const auto value : cpu_values)
        std::cout << value << std::endl;
    }
  for (auto request : requests)
    {
      MPI_Status status;
      MPI_Wait(&request, &status);
      int number_amount;
      MPI_Get_count(&status, MPI_DOUBLE, &number_amount);
      std::cout << "received " << number_amount << " elements from "
                << status.MPI_SOURCE << " with tag " << status.MPI_TAG
                << std::endl;
    }

  {
    std::vector<double> cpu_values(my_total_size);
    Utilities::CUDA::copy_to_host(device_memory_pointer, cpu_values);
    for (unsigned int j = 0; j < cpu_values.size(); ++j)
      std::cout << device_memory_pointer + j << " : " << cpu_values[j]
                << std::endl;
  }
}



int
main(int argc, char **argv)
{
  Utilities::CUDA::Handle cuda_handle;
  // By default, all the ranks will try to access the device 0. This is fine if
  // we have one rank per node _and_ one gpu per node. If we have multiple GPUs
  // on one node, we need each process to access a different GPU. We assume that
  // each node has the same number of GPUs.
  int         n_devices       = 0;
  cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
  AssertCuda(cuda_error_code);
  int device_id   = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK")) % n_devices;
  cuda_error_code = cudaSetDevice(device_id);
  AssertCuda(cuda_error_code);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  unsigned int myid = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  deallog.push(Utilities::int_to_string(myid));

  if (myid == 0)
    {
      initlog();
      deallog << std::setprecision(4);

      test0();
    }
  else
    test1();
}
