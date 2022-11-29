// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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


#ifndef dealii_memory_space_data_h
#define dealii_memory_space_data_h

#include <deal.II/base/config.h>

#include <deal.II/base/cuda.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/kokkos.h>

#include <functional>
#include <memory>

#  include <Kokkos_Core.hpp>

DEAL_II_NAMESPACE_OPEN

/**
 */
namespace MemorySpace
{
  /**
   * Structure which stores data on the host or the device depending on the
   * template parameter @p MemorySpace. The data is copied into the structure
   * which then owns the data and will release the memory when the destructor is
   * called.
   */
  template <typename T, typename MemorySpace>
  struct MemorySpaceData
  {
    MemorySpaceData();

    /**
     * Copy the active data (values for Host and values_dev for Device) to @p begin.
     * If the data is on the device it is moved to the host.
     */
    void
    copy_to(T *begin, const std::size_t n_elements);

    /**
     * Copy the data in @p begin to the active data of the structure (values for
     * Host and values_dev for Device). The pointer @p begin must be on the host.
     */
    void
    copy_from(const T *begin, const std::size_t n_elements);

    /** 
     * Kokkos View to the data on the host
     */
    Kokkos::View<T*, Kokkos::HostSpace> values;

    /**
     * Kokkos View to the data on the device
     */
    Kokkos::View<T *, MemorySpace> values_dev;

    /**
     * Pointers to the data of the processes sharing the same memory.
     */
    std::vector<ArrayView<const T>> values_sm;
  };



  /**
   * Swap function similar to std::swap.
   */
  template <typename T, typename MemorySpace>
  inline void
  swap(MemorySpaceData<T, MemorySpace> &u, MemorySpaceData<T, MemorySpace> &v);


#ifndef DOXYGEN

  template <typename T, typename MemorySpace>
  MemorySpaceData<T, MemorySpace>::MemorySpaceData()
    : values((dealii::Impl::ensure_kokkos_initialized(),
              Kokkos::View<T *, Kokkos::HostSpace>("host data", 0))),
     values_dev(Kokkos::View<T *, MemorySpace>("memoryspace data", 0))
  {}



  template <typename T, typename MemorySpace>
  bool
  MemorySpaceData<T, MemorySpace>::has_data_on_host()
  {
    return values.extent(0) > 0;
  }



  template <typename T, typename MemorySpace>
  T *
  MemorySpaceData<T, MemorySpace>::data()
  {
    return values.data();
  }



  template <typename T, typename MemorySpace>
  const T *
  MemorySpaceData<T, MemorySpace>::data() const
  {
    return values.data();
  }


  template <typename T, typename MemorySpace>
  T &
  MemorySpaceData<T, MemorySpace>::operator()(const unsigned int i)
  {
    return values(i);
  }



  template <typename T, typename MemorySpace>
  const T &
  MemorySpaceData<T, MemorySpace>::operator()(const unsigned int i) const
  {
    return values(i);
  }



  template <typename T, typename MemorySpace>
  void
  MemorySpaceData<T, MemorySpace>::copy_to(T *               begin,
                                           const std::size_t n_elements)
  {
    Assert(n_elements <= values.extent(0),
           ExcMessage("n_elements greater than the size of values."));
    using ExecutionSpace = typename MemorySpace::execution_space;
    Kokkos::
      View<T *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        begin_view(begin, n_elements);
    Kokkos::deep_copy(
      ExecutionSpace{},
      begin_view,
      Kokkos::subview(values, Kokkos::make_pair(std::size_t(0), n_elements)));
    ExecutionSpace{}.fence();
  }



  template <typename T, typename MemorySpace>
  void
  MemorySpaceData<T, MemorySpace>::copy_from(const T *         begin,
                                             const std::size_t n_elements)
  {
    Assert(n_elements <= values.extent(0),
           ExcMessage("n_elements greater than the size of values."));
    using ExecutionSpace = typename MemorySpace::execution_space;
    Kokkos::View<const T *,
                 Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      begin_view(begin, n_elements);
    Kokkos::deep_copy(
      ExecutionSpace{},
      Kokkos::subview(values, Kokkos::make_pair(std::size_t(0), n_elements)),
      begin_view);
    ExecutionSpace{}.fence();
  }



  /**
   * Swap function similar to std::swap.
   */
  template <typename T, typename MemorySpace>
  inline void
  swap(MemorySpaceData<T, MemorySpace> &u, MemorySpaceData<T, MemorySpace> &v)
  {
    auto u_copy = Kokkos::create_mirror(Kokkos::WithoutInitializing, u);
    typename MemorySpace::execution_space exec_space;
    // The first two calls to Kokkos::deep_copy are asynchronous. The last call
    // will wait for the three deep_copy to be done before returning.
    Kokkos::deep_copy(exec_space, u_copy, u);
    Kokkos::deep_copy(exec_space, u, v);
    Kokkos::deep_copy(v, u_copy);
  }

#endif

} // namespace MemorySpace

DEAL_II_NAMESPACE_CLOSE

#endif
