// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2020 by the deal.II authors
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


#ifndef dealii_memory_space_h
#define dealii_memory_space_h

#include <deal.II/base/config.h>

#include <Kokkos_Core.hpp>

DEAL_II_NAMESPACE_OPEN

/**
 */
namespace MemorySpace
{
  /**
   * Structure describing Host memory space.
   */
  using Host = ::Kokkos::HostSpace;

  /**
   * Structure describing Device memory space.
   */
  using Device = ::Kokkos::DefaultExecutionSpace::memory_space;

#ifdef DEAL_II_WITH_CUDA
  /**
   * Structure describing CUDA memory space.
   */
  using CUDA = ::Kokkos::CudaSpace;
#endif

} // namespace MemorySpace

DEAL_II_NAMESPACE_CLOSE

#endif
