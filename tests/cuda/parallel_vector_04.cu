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

#include <deal.II/lac/la_parallel_vector.templates.h>
#include <deal.II/lac/read_write_vector.h>

#include <iostream>
#include <vector>

#include "../tests.h"


void
test_cuda()
{
  const unsigned int my_id = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int my_owned_size = 2; // locally_owned_array.size();
  const unsigned int my_ghost_size =
    (my_id == 0) ? 0 : 1; // ghost_array.size(); /// TODO
  const unsigned int my_total_size         = my_owned_size + my_ghost_size;
  double *const      device_memory_pointer = [](const std::size_t size) {
    double *device_ptr;
    Utilities::CUDA::malloc(device_ptr, size);
    return device_ptr;
  }(my_total_size);
  {
    std::vector<double> cpu_values(my_total_size);
    for (unsigned int i = 0; i < my_total_size; ++i)
      {
        const unsigned int offset = (my_id == 0) ? 10 : 100;
        cpu_values[i]             = i + offset;
      }
    Utilities::CUDA::copy_to_dev(cpu_values, device_memory_pointer);
  }
  {
    std::vector<double> cpu_values(my_total_size);
    Utilities::CUDA::copy_to_host(device_memory_pointer, cpu_values);
    std::cout << "NEW previous" << std::endl;
    for (unsigned int j = 0; j < cpu_values.size(); ++j)
      std::cout << device_memory_pointer + j << " : " << cpu_values[j]
                << std::endl;
    std::cout << "NEW end" << std::endl;
  }

  double *const device_owned_pointer = device_memory_pointer;
  double *const device_ghost_pointer = device_memory_pointer + my_owned_size;

  const unsigned int n_import_targets = (my_id == 0) ? 1 : 0;
  const unsigned int n_ghost_targets  = (my_id == 0) ? 0 : 1;

  std::vector<MPI_Request> new_requests(
    0); // n_import_targets + n_ghost_targets);

  if (n_ghost_targets > 0)
    {
      cudaPointerAttributes attributes;
      const cudaError_t     cuda_error =
        cudaPointerGetAttributes(&attributes, device_ghost_pointer);
      std::cout << "NEW Receiving on device: " << attributes.device << " at "
                << device_ghost_pointer << std::endl;
      MPI_Status status;
      const int  ierr = MPI_Recv(device_ghost_pointer,
                                1,
                                MPI_DOUBLE,
                                0,    // source
                                1000, // channel
                                MPI_COMM_WORLD,
                                &status); //&new_requests[0]);
      int        number_amount;
      MPI_Get_count(&status, MPI_DOUBLE, &number_amount);
      std::cout << "NEW received " << number_amount << " elements from "
                << status.MPI_SOURCE << " with tag " << status.MPI_TAG
                << std::endl;
      AssertThrowMPI(ierr);
    }

  if (n_import_targets > 0)
    {
      cudaPointerAttributes attributes;
      const cudaError_t     cuda_error =
        cudaPointerGetAttributes(&attributes, device_owned_pointer);
      std::cout << "NEW Sending from device: " << attributes.device << " at "
                << device_owned_pointer << std::endl;
      const int ierr = MPI_Send(device_owned_pointer,
                                 1,
                                 MPI_DOUBLE,
                                 1,    // destination,
                                 1000, // channel,
                                 MPI_COMM_WORLD/*,
                                 &new_requests[0]*/);
      AssertThrowMPI(ierr);

      std::vector<double> cpu_values(1);
      Utilities::CUDA::copy_to_host(device_owned_pointer, cpu_values);
      for (const auto value : cpu_values)
        std::cout << value << std::endl;
    }
  for (auto request : new_requests)
    {
      MPI_Status status;
      MPI_Wait(&request, &status);
      int number_amount;
      MPI_Get_count(&status, MPI_DOUBLE, &number_amount);
      std::cout << "NEW received " << number_amount << " elements from "
                << status.MPI_SOURCE << " with tag " << status.MPI_TAG
                << std::endl;
    }

  std::vector<double> cpu_values(my_total_size);
  Utilities::CUDA::copy_to_host(device_memory_pointer, cpu_values);
  std::cout << "NEW" << std::endl;
  for (unsigned int j = 0; j < cpu_values.size(); ++j)
    std::cout << device_memory_pointer + j << " : " << cpu_values[j]
              << std::endl;
  std::cout << "NEW end" << std::endl;
  const cudaError_t error_code = cudaFree(device_memory_pointer);
  AssertCuda(error_code);
}


__global__ void
set_value(double *values_dev, unsigned int index, double val)
{
  values_dev[index] = val;
}

void
test()
{
  std::cout << "1 test" << std::endl;
  test_cuda();
  unsigned int myid    = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  unsigned int numproc = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  const unsigned int my_owned_size = 2; // locally_owned_array.size();
  const unsigned int my_ghost_size =
    (myid == 0) ? 0 : 1; // ghost_array.size(); /// TODO
  const unsigned int my_total_size         = my_owned_size + my_ghost_size;
  double *const      device_memory_pointer = [](const std::size_t size) {
    double *device_ptr;
    Utilities::CUDA::malloc(device_ptr, size);
    return device_ptr;
  }(my_total_size);

  {
    int num_elements = 10;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int *data = (int *)malloc(sizeof(int) * num_elements);
    assert(data != NULL);

    /*for (unsigned int i = 0; i < num_elements; ++i)
      {
        data[i] = i + world_rank;
        std::cout << data[i] << std::endl;
      }*/

    // Time MPI_Bcast
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    /*for (unsigned int i = 0; i < num_elements; ++i)
      {
        std::cout << data[i] << std::endl;
      }*/
  }

  if (myid == 0)
    deallog << "numproc=" << numproc << std::endl;


  // each processor owns 2 indices and all
  // are ghosting element 1 (the second)
  IndexSet local_owned(numproc * 2);
  local_owned.add_range(myid * 2, myid * 2 + 2);
  IndexSet local_relevant(numproc * 2);
  local_relevant = local_owned;
  local_relevant.add_range(1, 2);

  std::cout << "2 test" << std::endl;
  test_cuda();

  const cudaError_t error_code = cudaFree(device_memory_pointer);
  AssertCuda(error_code);

  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> v(
    local_owned, local_relevant, MPI_COMM_WORLD);

  std::cout << "3 test" << std::endl;
  test_cuda();

  // set local values and check them
  LinearAlgebra::ReadWriteVector<double> rw_vector(local_owned);
  rw_vector(myid * 2)     = myid * 2.0;
  rw_vector(myid * 2 + 1) = myid * 2.0 + 1.0;

  v.import(rw_vector, VectorOperation::insert);
  v *= 2.0;

  std::cout << "4 test" << std::endl;
  // test_cuda();

  rw_vector.import(v, VectorOperation::insert);
  AssertThrow(rw_vector(myid * 2) == myid * 4.0, ExcInternalError());
  AssertThrow(rw_vector(myid * 2 + 1) == myid * 4.0 + 2.0, ExcInternalError());

  // set ghost dof on remote process, no compress called. Since we don't want to
  // call compress we cannot use import
  auto partitioner = v.get_partitioner();
  if (myid > 0)
    {
      unsigned int local_index = partitioner->global_to_local(1);
      double *     values_dev  = v.get_values();
      set_value<<<1, 1>>>(values_dev, local_index, 7);
    }

  std::cout << "5 test" << std::endl;
  // test_cuda();


  unsigned int        allocated_size = local_relevant.n_elements();
  std::vector<double> v_host(allocated_size);
  Utilities::CUDA::copy_to_host(v.get_values(), v_host);

  std::cout << "6 test" << std::endl;
  // test_cuda();


  AssertThrow(v_host[partitioner->global_to_local(myid * 2)] == myid * 4.0,
              ExcInternalError());
  AssertThrow(v_host[partitioner->global_to_local(myid * 2 + 1)] ==
                myid * 4.0 + 2.0,
              ExcInternalError());

  std::cout << "7 test" << std::endl;
  // test_cuda();


  if (myid > 0)
    AssertThrow(v_host[partitioner->global_to_local(1)] == 7.0,
                ExcInternalError());

  std::cout << "8 test" << std::endl;
  // test_cuda();


  // reset to zero
  v = 0;

  std::cout << "9 test" << std::endl;
  // test_cuda();

  Utilities::CUDA::copy_to_host(v.get_values(), v_host);
  AssertThrow(v_host[partitioner->global_to_local(myid * 2)] == 0.,
              ExcInternalError());
  AssertThrow(v_host[partitioner->global_to_local(myid * 2 + 1)] == 0.,
              ExcInternalError());

  std::cout << "10 test" << std::endl;
  // test_cuda();


  // check that everything remains zero also
  // after compress
  v.compress(VectorOperation::add);

  std::cout << "11 test" << std::endl;
  // test_cuda();


  Utilities::CUDA::copy_to_host(v.get_values(), v_host);
  AssertThrow(v_host[partitioner->global_to_local(myid * 2)] == 0.,
              ExcInternalError());
  AssertThrow(v_host[partitioner->global_to_local(myid * 2 + 1)] == 0.,
              ExcInternalError());

  std::cout << "12 test" << std::endl;
  // test_cuda();

  // set element 1 on owning process to
  // something nonzero
  if (myid == 0)
    {
      unsigned int local_index = partitioner->global_to_local(1);
      double *     values_dev  = v.get_values();
      set_value<<<1, 1>>>(values_dev, local_index, 2);
      Utilities::CUDA::copy_to_host(v.get_values(), v_host);
      AssertThrow(v_host[partitioner->global_to_local(1)] == 2.,
                  ExcInternalError());
    }
  if (myid > 0)
    {
      Utilities::CUDA::copy_to_host(v.get_values(), v_host);
      AssertThrow(v_host[partitioner->global_to_local(1)] == 0.,
                  ExcInternalError());
    }

  std::cout << "13 test" << std::endl;
  // test_cuda();

  std::cout << "Start" << std::endl;
  v.print(std::cout);

  // test_cuda();
  // check that all processors get the correct
  // value again, and that it is erased by
  // operator=
  v.update_ghost_values();
  v.print(std::cout);


  std::cout << "14 test" << std::endl;
  // test_cuda();

  Utilities::CUDA::copy_to_host(v.get_values(), v_host);
  AssertThrow(v_host[partitioner->global_to_local(1)] == 2.,
              ExcInternalError());

  v = 0;
  Utilities::CUDA::copy_to_host(v.get_values(), v_host);
  AssertThrow(v_host[partitioner->global_to_local(1)] == 0.,
              ExcInternalError());

  if (myid == 0)
    deallog << "OK" << std::endl;
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

      test();
    }
  else
    {
      test();
    }
}
