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


// check global reduction operation (norms, operator==, operator!=) on
// parallel vector

#include <deal.II/base/cuda.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/read_write_vector.h>

#include <iostream>
#include <vector>

#include "../tests.h"


void
test()
{
  unsigned int myid    = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  unsigned int numproc = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  if (myid == 0)
    deallog << "numproc=" << numproc << std::endl;


  // each processor owns 2 indices and all processors are
  // ghosting element 1 (the second)
  IndexSet local_owned(std::min(16U, numproc * 2));
  local_owned.add_range(myid * 2, myid * 2 + 2);
  IndexSet local_relevant(numproc * 2);
  local_relevant = local_owned;
  local_relevant.add_range(1, 2);

  LinearAlgebra::distributed::Vector<double, CUDA> v(local_owned,
                                                     local_owned,
                                                     MPI_COMM_WORLD);

  // set local values
  LinearAlgebra::ReadWriteVector<double> rw_vector(local_owned);
  {
    rw_vector(myid * 2)     = myid * 2.0;
    rw_vector(myid * 2 + 1) = myid * 2.0 + 1.0;
  }
  v.import(rw_vector, VectorOperation::insert);
  v *= 2.0;
  {
    rw_vector.import(v, VectorOperation::insert);
    AssertThrow(rw_vector(myid * 2) == myid * 4.0, ExcInternalError());
    AssertThrow(rw_vector(myid * 2 + 1) == myid * 4.0 + 2.0,
                ExcInternalError());
  }

  // check l2 norm
  {
    const double l2_norm = v.l2_norm();
    if (myid == 0)
      deallog << "l2 norm: " << l2_norm << std::endl;
  }

  // check l1 norm
  {
    const double l1_norm = v.l1_norm();
    if (myid == 0)
      deallog << "l1 norm: " << l1_norm << std::endl;
  }

  // check linfty norm
  {
    const double linfty_norm = v.linfty_norm();
    if (myid == 0)
      deallog << "linfty norm: " << linfty_norm << std::endl;
  }

  // check mean value (should be equal to l1
  // norm divided by vector size here since we
  // have no negative entries)
  {
    const double mean = v.mean_value();
    if (myid == 0)
      deallog << "Mean value: " << mean << std::endl;

    Assert(std::fabs(mean * v.size() - v.l1_norm()) < 1e-15,
           ExcInternalError());
  }
  // check inner product
  {
    const double norm_sqr = v.l2_norm() * v.l2_norm();
    AssertThrow(std::fabs(v * v - norm_sqr) < 1e-15, ExcInternalError());
    LinearAlgebra::distributed::Vector<double, CUDA> v2;
    v2 = v;
    AssertThrow(std::fabs(v2 * v - norm_sqr) < 1e-15, ExcInternalError());

    const double inner_prod = v * v2;
    if (myid == 0)
      deallog << "Inner product: " << inner_prod << std::endl;
  }

  // check all_zero
  {
    bool allzero = v.all_zero();
    if (myid == 0)
      deallog << " v==0 ? " << allzero << std::endl;
    LinearAlgebra::distributed::Vector<double, CUDA> v2;
    v2.reinit(v);
    allzero = v2.all_zero();
    if (myid == 0)
      deallog << " v2==0 ? " << allzero << std::endl;

    v2.import(rw_vector, VectorOperation::insert);
    allzero = v2.all_zero();
    if (myid == 0)
      deallog << " v2==0 ? " << allzero << std::endl;
  }

  if (myid == 0)
    deallog << "OK" << std::endl;
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  unsigned int myid = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  deallog.push(Utilities::int_to_string(myid));

  Utilities::CUDA::Handle cuda_handle;
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


  if (myid == 0)
    {
      initlog();
      deallog << std::setprecision(4);

      test();
    }
  else
    test();
}
