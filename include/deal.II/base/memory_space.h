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

#ifdef DEAL_II_WITH_KOKKOS_BACKEND
#  include <Kokkos_Core.hpp>
#endif

DEAL_II_NAMESPACE_OPEN

/**
 */
namespace MemorySpace
{
#ifdef DEAL_II_WITH_KOKKOS_BACKEND
  /**
   * Structure describing Host memory space.
   */
  using Host = ::Kokkos::HostSpace;
#else
  /**
   * Structure describing Host memory space.
   */
  struct Host
  {};
#endif


#ifdef DEAL_II_WITH_KOKKOS_BACKEND
  /**
   * Structure describing CUDA memory space.
   */
  using CUDA = ::Kokkos::CudaSpace;
#else
  /**
   * Structure describing CUDA memory space.
   */
  struct CUDA
  {};
#endif

} // namespace MemorySpace

DEAL_II_NAMESPACE_CLOSE

#endif
