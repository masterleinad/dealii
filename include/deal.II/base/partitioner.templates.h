// ---------------------------------------------------------------------
//
// Copyright (C) 2017 - 2018 by the deal.II authors
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

#ifndef dealii_partitioner_templates_h
#define dealii_partitioner_templates_h

#include <deal.II/base/config.h>

#include <deal.II/base/cuda_size.h>
#include <deal.II/base/partitioner.h>

#include <deal.II/lac/cuda_kernels.templates.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <type_traits>


DEAL_II_NAMESPACE_OPEN

namespace Utilities
{
  namespace MPI
  {
#ifndef DOXYGEN

#  ifdef DEAL_II_WITH_MPI

    template <typename Number, typename MemorySpaceType>
    void
    Partitioner::export_to_ghosted_array_start(
      const unsigned int             communication_channel,
      const ArrayView<const Number> &locally_owned_array,
      const ArrayView<Number> &      temporary_storage,
      const ArrayView<Number> &      ghost_array,
      std::vector<MPI_Request> &     requests) const
    {
      AssertDimension(temporary_storage.size(), n_import_indices());
      Assert(ghost_array.size() == n_ghost_indices() ||
               ghost_array.size() == n_ghost_indices_in_larger_set,
             ExcGhostIndexArrayHasWrongSize(ghost_array.size(),
                                            n_ghost_indices(),
                                            n_ghost_indices_in_larger_set));

      const unsigned int n_import_targets = import_targets_data.size();
      std::cout << "n_import_targets: " << n_import_targets << std::endl;
      const unsigned int n_ghost_targets = ghost_targets_data.size();
      std::cout << "n_ghost_targets: " << n_ghost_targets << std::endl;
      std::cout << "locally_owned_array starts at "
                << locally_owned_array.data() << std::endl;
      std::cout << "temporary_storage starts at " << temporary_storage.data()
                << std::endl;
      std::cout << "ghost_array starts at " << ghost_array.data() << std::endl;
      std::cout << "pointer diff "
                << ghost_array.data() - locally_owned_array.data() << std::endl;
      std::cout << "sizeof(Number): " << sizeof(Number) << std::endl;

      if (n_import_targets > 0)
        AssertDimension(locally_owned_array.size(), local_size());

      Assert(requests.size() == 0,
             ExcMessage("Another operation seems to still be running. "
                        "Call update_ghost_values_finish() first."));

// TODO start
#    if defined(DEAL_II_COMPILER_CUDA_AWARE) && \
      defined(DEAL_II_WITH_CUDA_AWARE_MPI)
      {
        const unsigned int my_owned_size = 2; // locally_owned_array.size();
        const unsigned int my_ghost_size = ghost_array.size(); /// TODO
        const unsigned int my_total_size = my_owned_size + my_ghost_size;
        double *const      device_memory_pointer = [](const std::size_t size) {
          double *device_ptr;
          Utilities::CUDA::malloc(device_ptr, size);
          return device_ptr;
        }(my_total_size);
        {
          std::vector<double> cpu_values(my_total_size);
          for (unsigned int i = 0; i < my_total_size; ++i)
            {
              cpu_values[i] = i + 10;
            }
          Utilities::CUDA::copy_to_dev(cpu_values, device_memory_pointer);
        }

        double *const device_owned_pointer = device_memory_pointer;
        double *const device_ghost_pointer =
          device_memory_pointer + my_owned_size;

        std::vector<MPI_Request> new_requests(n_import_targets +
                                              n_ghost_targets);

        if (n_ghost_targets > 0)
          {
            const int ierr = MPI_Irecv(device_ghost_pointer,
                                       1,
                                       MPI_DOUBLE,
                                       0,    // source
                                       1000, // channel
                                       MPI_COMM_WORLD,
                                       &new_requests[0]);
            AssertThrowMPI(ierr);
          }

        if (n_import_targets > 0)
          {
            const int ierr = MPI_Isend(device_owned_pointer,
                                       1,
                                       MPI_DOUBLE,
                                       1,    // destination,
                                       1000, // channel,
                                       communicator,
                                       &new_requests[0]);
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
        Utilities::CUDA::copy_to_host(device_owned_pointer, cpu_values);
        std::cout << "NEW" << std::endl;
        for (unsigned int j = 0; j < cpu_values.size(); ++j)
          std::cout << device_memory_pointer + j << " : " << cpu_values[j]
                    << std::endl;
        std::cout << "NEW end" << std::endl;
      }
#    endif
      // TODO end


      // Need to send and receive the data. Use non-blocking communication,
      // where it is usually less overhead to first initiate the receive and
      // then actually send the data
      requests.resize(n_import_targets + n_ghost_targets);

      // as a ghost array pointer, put the data at the end of the given ghost
      // array in case we want to fill only a subset of the ghosts so that we
      // can move data to the right position in a forward loop in the _finish
      // function.
      AssertIndexRange(n_ghost_indices(), n_ghost_indices_in_larger_set + 1);
      const bool use_larger_set =
        (n_ghost_indices_in_larger_set > n_ghost_indices() &&
         ghost_array.size() == n_ghost_indices_in_larger_set);
      Number *ghost_array_ptr =
        use_larger_set ? ghost_array.data() + n_ghost_indices_in_larger_set -
                           n_ghost_indices() :
                         ghost_array.data();

      for (unsigned int i = 0; i < n_ghost_targets; i++)
        {
          // allow writing into ghost indices even though we are in a
          // const function
          const int ierr =
            MPI_Irecv(ghost_array_ptr,
                      ghost_targets_data[i].second,
                      MPI_DOUBLE,
                      ghost_targets_data[i].first,
                      ghost_targets_data[i].first + communication_channel,
                      communicator,
                      &requests[i]);
          AssertThrowMPI(ierr);
          std::cout << "Receiving " << ghost_targets_data[i].second
                    << " values from " << ghost_targets_data[i].first
                    << " on channel "
                    << ghost_targets_data[i].first + communication_channel
                    << " to be stored at " << ghost_array_ptr << std::endl;
          ghost_array_ptr += ghost_targets_data[i].second;
        }


      /*        // TODO start
      #    if defined(DEAL_II_COMPILER_CUDA_AWARE) && \
            defined(DEAL_II_WITH_CUDA_AWARE_MPI)
            if (n_import_targets > 0)
              {
                const int ierr = MPI_Isend(device_owned_pointer,
                                           1,
                                           MPI_DOUBLE,
                                           1,    // destination,
                                           1000, // channel,
                                           communicator,
                                           &new_requests[0]);
                AssertThrowMPI(ierr);

                std::vector<Number> cpu_values(1);
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
                std::cout << "NEW received " << number_amount << " elements from
      "
                          << status.MPI_SOURCE << " with tag " << status.MPI_TAG
                          << std::endl;
              }

            std::vector<Number> cpu_values(locally_owned_array.size() +
                                           ghost_array.size());
            Utilities::CUDA::copy_to_host(device_owned_pointer, cpu_values);
            std::cout << "NEW" << std::endl;
            for (unsigned int j = 0;
                 j < locally_owned_array.size() + ghost_array.size();
                 ++j)
              std::cout << device_memory_pointer + j << " : " << cpu_values[j]
                        << std::endl;
            std::cout << "NEW end" << std::endl;
      #    endif
            // TODO end*/

      Number *temp_array_ptr = temporary_storage.data();
      for (unsigned int i = 0; i < n_import_targets; i++)
        {
          // copy the data to be sent to the import_data field
          std::vector<std::pair<unsigned int, unsigned int>>::const_iterator
            my_imports = import_indices_data.begin() +
                         import_indices_chunks_by_rank_data[i],
            end_my_imports = import_indices_data.begin() +
                             import_indices_chunks_by_rank_data[i + 1];
          unsigned int index = 0;
          for (; my_imports != end_my_imports; ++my_imports)
            {
              const unsigned int chunk_size =
                my_imports->second - my_imports->first;
              std::cout << "Copying elements [" << my_imports->first << ","
                        << my_imports->second << "] "
                        << "from "
                        << locally_owned_array.data() + my_imports->first
                        << " to " << temp_array_ptr + index << std::endl;
#    if defined(DEAL_II_COMPILER_CUDA_AWARE) && \
      defined(DEAL_II_WITH_CUDA_AWARE_MPI)
              if (std::is_same<MemorySpaceType, MemorySpace::CUDA>::value)
                {
                  const cudaError_t cuda_error_code =
                    cudaMemcpy(temp_array_ptr + index,
                               locally_owned_array.data() + my_imports->first,
                               chunk_size * sizeof(Number),
                               cudaMemcpyDeviceToDevice);
                  AssertCuda(cuda_error_code);
                  std::cout << "CUDA" << std::endl;
                }
              else
#    endif
                {
                  std::memcpy(temp_array_ptr + index,
                              locally_owned_array.data() + my_imports->first,
                              chunk_size * sizeof(Number));
                  std::cout << "CPU" << std::endl;
                }
              index += chunk_size;
            }

          AssertDimension(index, import_targets_data[i].second);

          Assert((std::is_same<Number, double>::value), ExcInternalError());

          // start the send operations
          const int ierr =
            MPI_Isend(locally_owned_array.data() /*temp_array_ptr*/,
                      import_targets_data[i].second,
                      MPI_DOUBLE,
                      import_targets_data[i].first,
                      my_pid + communication_channel,
                      communicator,
                      &requests[n_ghost_targets + i]);
          AssertThrowMPI(ierr);
          std::cout << "Sending " << import_targets_data[i].second
                    << "elements: \n";
          {
#    if !(defined(DEAL_II_COMPILER_CUDA_AWARE) && \
          defined(DEAL_II_WITH_CUDA_AWARE_MPI))

            for (unsigned int j = 0; j < import_targets_data[i].second; ++j)
              std::cout << temp_array_ptr[j] << std::endl;
#    else
            std::vector<Number> cpu_values(import_targets_data[i].second);
            Utilities::CUDA::copy_to_host(
              locally_owned_array.data() /*temp_array_ptr*/, cpu_values);
            for (const auto value : cpu_values)
              std::cout << value << std::endl;
#    endif
          }
          std::cout << "to " << import_targets_data[i].first << " on channel "
                    << my_pid + communication_channel << std::endl;

          temp_array_ptr += import_targets_data[i].second;
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
#    if !(defined(DEAL_II_COMPILER_CUDA_AWARE) && \
          defined(DEAL_II_WITH_CUDA_AWARE_MPI))

      for (unsigned int j = 0;
           j < locally_owned_array.size() + ghost_array.size();
           ++j)
        std::cout << locally_owned_array.data() + j << " : "
                  << *(locally_owned_array.data() + j) << std::endl;
#    else
      {
        std::vector<Number> cpu_values(locally_owned_array.size() +
                                       ghost_array.size());
        Utilities::CUDA::copy_to_host(locally_owned_array.data(), cpu_values);
        for (unsigned int j = 0;
             j < locally_owned_array.size() + ghost_array.size();
             ++j)
          std::cout << locally_owned_array.data() + j << " : " << cpu_values[j]
                    << std::endl;
      }
#    endif
      MPI_Barrier(MPI_COMM_WORLD);
    }



    template <typename Number>
    void
    Partitioner::export_to_ghosted_array_finish(
      const ArrayView<Number> & ghost_array,
      std::vector<MPI_Request> &requests) const
    {
      Assert(ghost_array.size() == n_ghost_indices() ||
               ghost_array.size() == n_ghost_indices_in_larger_set,
             ExcGhostIndexArrayHasWrongSize(ghost_array.size(),
                                            n_ghost_indices(),
                                            n_ghost_indices_in_larger_set));

      // wait for both sends and receives to complete, even though only
      // receives are really necessary. this gives (much) better performance
      AssertDimension(ghost_targets().size() + import_targets().size(),
                      requests.size());
      if (requests.size() > 0)
        {
          const int ierr =
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
          AssertThrowMPI(ierr);
        }
      requests.resize(0);

      // in case we only sent a subset of indices, we now need to move the data
      // to the correct positions and delete the old content
      if (n_ghost_indices_in_larger_set > n_ghost_indices() &&
          ghost_array.size() == n_ghost_indices_in_larger_set)
        {
          unsigned int offset =
            n_ghost_indices_in_larger_set - n_ghost_indices();
          // must copy ghost data into extended ghost array
          for (std::vector<std::pair<unsigned int, unsigned int>>::
                 const_iterator my_ghosts = ghost_indices_subset_data.begin();
               my_ghosts != ghost_indices_subset_data.end();
               ++my_ghosts)
            if (offset > my_ghosts->first)
              for (unsigned int j = my_ghosts->first; j < my_ghosts->second;
                   ++j, ++offset)
                {
                  ghost_array[j]      = ghost_array[offset];
                  ghost_array[offset] = Number();
                }
            else
              {
                AssertDimension(offset, my_ghosts->first);
                break;
              }
        }
    }



    template <typename Number>
    void
    Partitioner::import_from_ghosted_array_start(
      const VectorOperation::values vector_operation,
      const unsigned int            communication_channel,
      const ArrayView<Number> &     ghost_array,
      const ArrayView<Number> &     temporary_storage,
      std::vector<MPI_Request> &    requests) const
    {
      AssertDimension(temporary_storage.size(), n_import_indices());
      Assert(ghost_array.size() == n_ghost_indices() ||
               ghost_array.size() == n_ghost_indices_in_larger_set,
             ExcGhostIndexArrayHasWrongSize(ghost_array.size(),
                                            n_ghost_indices(),
                                            n_ghost_indices_in_larger_set));

      (void)vector_operation;

      // nothing to do for insert (only need to zero ghost entries in
      // compress_finish()). in debug mode we want to check consistency of the
      // inserted data, therefore the communication is still initialized.
      // Having different code in debug and optimized mode is somewhat
      // dangerous, but it really saves communication so do it anyway
#    ifndef DEBUG
      if (vector_operation == VectorOperation::insert)
        return;
#    endif

      // nothing to do when we neither have import
      // nor ghost indices.
      if (n_ghost_indices() == 0 && n_import_indices() == 0)
        return;

      const unsigned int n_import_targets = import_targets_data.size();
      const unsigned int n_ghost_targets  = ghost_targets_data.size();

      Assert(requests.size() == 0,
             ExcMessage("Another compress operation seems to still be running. "
                        "Call compress_finish() first."));

      // Need to send and receive the data. Use non-blocking communication,
      // where it is generally less overhead to first initiate the receive and
      // then actually send the data

      // set channels in different range from update_ghost_values channels
      const unsigned int channel = communication_channel + 401;
      requests.resize(n_import_targets + n_ghost_targets);

      // initiate the receive operations
      Number *temp_array_ptr = temporary_storage.data();
      for (unsigned int i = 0; i < n_import_targets; i++)
        {
          AssertThrow(
            static_cast<std::size_t>(import_targets_data[i].second) *
                sizeof(Number) <
              static_cast<std::size_t>(std::numeric_limits<int>::max()),
            ExcMessage("Index overflow: Maximum message size in MPI is 2GB. "
                       "The number of ghost entries times the size of 'Number' "
                       "exceeds this value. This is not supported."));
          const int ierr =
            MPI_Irecv(temp_array_ptr,
                      import_targets_data[i].second * sizeof(Number),
                      MPI_BYTE,
                      import_targets_data[i].first,
                      import_targets_data[i].first + channel,
                      communicator,
                      &requests[i]);
          AssertThrowMPI(ierr);
          temp_array_ptr += import_targets_data[i].second;
        }

      // initiate the send operations

      // in case we want to import only from a subset of the ghosts we want to
      // move the data to send to the front of the array
      AssertIndexRange(n_ghost_indices(), n_ghost_indices_in_larger_set + 1);
      Number *ghost_array_ptr = ghost_array.data();
      for (unsigned int i = 0; i < n_ghost_targets; i++)
        {
          // in case we only sent a subset of indices, we now need to move the
          // data to the correct positions and delete the old content
          if (n_ghost_indices_in_larger_set > n_ghost_indices() &&
              ghost_array.size() == n_ghost_indices_in_larger_set)
            {
              std::vector<std::pair<unsigned int, unsigned int>>::const_iterator
                my_ghosts = ghost_indices_subset_data.begin() +
                            ghost_indices_subset_chunks_by_rank_data[i],
                end_my_ghosts = ghost_indices_subset_data.begin() +
                                ghost_indices_subset_chunks_by_rank_data[i + 1];
              unsigned int offset = 0;
              for (; my_ghosts != end_my_ghosts; ++my_ghosts)
                if (ghost_array_ptr + offset !=
                    ghost_array.data() + my_ghosts->first)
                  for (unsigned int j = my_ghosts->first; j < my_ghosts->second;
                       ++j, ++offset)
                    {
                      ghost_array_ptr[offset] = ghost_array[j];
                      ghost_array[j]          = Number();
                    }
                else
                  offset += my_ghosts->second - my_ghosts->first;
              AssertDimension(offset, ghost_targets_data[i].second);
            }

          AssertThrow(
            static_cast<std::size_t>(ghost_targets_data[i].second) *
                sizeof(Number) <
              static_cast<std::size_t>(std::numeric_limits<int>::max()),
            ExcMessage("Index overflow: Maximum message size in MPI is 2GB. "
                       "The number of ghost entries times the size of 'Number' "
                       "exceeds this value. This is not supported."));
          const int ierr =
            MPI_Isend(ghost_array_ptr,
                      ghost_targets_data[i].second * sizeof(Number),
                      MPI_BYTE,
                      ghost_targets_data[i].first,
                      this_mpi_process() + channel,
                      communicator,
                      &requests[n_import_targets + i]);
          AssertThrowMPI(ierr);

          ghost_array_ptr += ghost_targets_data[i].second;
        }
    }



    namespace internal
    {
      // In the import_from_ghosted_array_finish we need to invoke abs() also
      // on unsigned data types, which is ill-formed on newer C++
      // standards. To avoid this, we use std::abs on default types but
      // simply return the number on unsigned types
      template <typename Number>
      typename std::enable_if<
        !std::is_unsigned<Number>::value,
        typename numbers::NumberTraits<Number>::real_type>::type
      get_abs(const Number a)
      {
        return std::abs(a);
      }

      template <typename Number>
      typename std::enable_if<std::is_unsigned<Number>::value, Number>::type
      get_abs(const Number a)
      {
        return a;
      }

      // In the import_from_ghosted_array_finish we might need to calculate the
      // maximal and minimal value for the given number type, which is not
      // straight forward for complex numbers. Therefore, comparison of complex
      // numbers is prohibited and throws an exception.
      template <typename Number>
      Number
      get_min(const Number a, const Number b)
      {
        return std::min(a, b);
      }

      template <typename Number>
      std::complex<Number>
      get_min(const std::complex<Number> a, const std::complex<Number>)
      {
        AssertThrow(false,
                    ExcMessage("VectorOperation::min not "
                               "implemented for complex numbers"));
        return a;
      }

      template <typename Number>
      Number
      get_max(const Number a, const Number b)
      {
        return std::max(a, b);
      }

      template <typename Number>
      std::complex<Number>
      get_max(const std::complex<Number> a, const std::complex<Number>)
      {
        AssertThrow(false,
                    ExcMessage("VectorOperation::max not "
                               "implemented for complex numbers"));
        return a;
      }
    } // namespace internal



    template <typename Number, typename MemorySpaceType>
    void
    Partitioner::import_from_ghosted_array_finish(
      const VectorOperation::values  vector_operation,
      const ArrayView<const Number> &temporary_storage,
      const ArrayView<Number> &      locally_owned_array,
      const ArrayView<Number> &      ghost_array,
      std::vector<MPI_Request> &     requests) const
    {
      AssertDimension(temporary_storage.size(), n_import_indices());
      Assert(ghost_array.size() == n_ghost_indices() ||
               ghost_array.size() == n_ghost_indices_in_larger_set,
             ExcGhostIndexArrayHasWrongSize(ghost_array.size(),
                                            n_ghost_indices(),
                                            n_ghost_indices_in_larger_set));

      // in optimized mode, no communication was started, so leave the
      // function directly (and only clear ghosts)
#    ifndef DEBUG
      if (vector_operation == VectorOperation::insert)
        {
          Assert(requests.empty(),
                 ExcInternalError(
                   "Did not expect a non-empty communication "
                   "request when inserting. Check that the same "
                   "vector_operation argument was passed to "
                   "import_from_ghosted_array_start as is passed "
                   "to import_from_ghosted_array_finish."));
#      ifdef DEAL_II_WITH_CXX17
          if constexpr (std::is_trivial<Number>::value)
#      else
          if (std::is_trivial<Number>::value)
#      endif
            std::memset(ghost_array.data(),
                        0,
                        sizeof(Number) * ghost_array.size());
          else
            std::fill(ghost_array.data(),
                      ghost_array.data() + ghost_array.size(),
                      0);
          return;
        }
#    endif

      // nothing to do when we neither have import nor ghost indices.
      if (n_ghost_indices() == 0 && n_import_indices() == 0)
        return;

      const unsigned int n_import_targets = import_targets_data.size();
      const unsigned int n_ghost_targets  = ghost_targets_data.size();

      if (vector_operation != dealii::VectorOperation::insert)
        AssertDimension(n_ghost_targets + n_import_targets, requests.size());
      // first wait for the receive to complete
      if (requests.size() > 0 && n_import_targets > 0)
        {
          AssertDimension(locally_owned_array.size(), local_size());
          const int ierr =
            MPI_Waitall(n_import_targets, requests.data(), MPI_STATUSES_IGNORE);
          AssertThrowMPI(ierr);

          {
#    if !(defined(DEAL_II_COMPILER_CUDA_AWARE) && \
          defined(DEAL_II_WITH_CUDA_AWARE_MPI))
            std::cout << "temporary storage" << std::endl;
            for (const auto value : temporary_storage)
              std::cout << value << std::endl;
            std::cout << "owned values" << std::endl;
            for (const auto value : locally_owned_array)
              std::cout << value << std::endl;
            std::cout << "ghost values" << std::endl;
            for (const auto value : ghost_array)
              std::cout << value << std::endl;
#    else
            std::vector<Number> cpu_values_temp(temporary_storage.size());
            Utilities::CUDA::copy_to_host(temporary_storage.data(),
                                          cpu_values_temp);
            std::cout << "temporary storage" << std::endl;
            for (const auto value : cpu_values_temp)
              std::cout << value << std::endl;

            std::vector<Number> cpu_values_owned(locally_owned_array.size());
            Utilities::CUDA::copy_to_host(locally_owned_array.data(),
                                          cpu_values_owned);
            std::cout << "owned values" << std::endl;
            for (const auto value : cpu_values_owned)
              std::cout << value << std::endl;

            std::vector<Number> cpu_values_ghost(ghost_array.size());
            Utilities::CUDA::copy_to_host(ghost_array.data(), cpu_values_ghost);
            std::cout << "ghost values" << std::endl;
            for (const auto value : cpu_values_ghost)
              std::cout << value << std::endl;
#    endif
          }

          const Number *read_position = temporary_storage.data();
#    if !(defined(DEAL_II_COMPILER_CUDA_AWARE) && \
          defined(DEAL_II_WITH_CUDA_AWARE_MPI))
          // If the operation is no insertion, add the imported data to the
          // local values. For insert, nothing is done here (but in debug mode
          // we assert that the specified value is either zero or matches with
          // the ones already present
          if (vector_operation == dealii::VectorOperation::add)
            for (const auto &import_range : import_indices_data)
              for (unsigned int j = import_range.first; j < import_range.second;
                   j++)
                {
                  locally_owned_array[j] += *read_position++;
                  std::cout << "locally_owned_array[" << j
                            << "]: " << locally_owned_array[j] << std::endl;
                }
          else if (vector_operation == dealii::VectorOperation::min)
            for (const auto &import_range : import_indices_data)
              for (unsigned int j = import_range.first; j < import_range.second;
                   j++)
                {
                  locally_owned_array[j] =
                    internal::get_min(*read_position, locally_owned_array[j]);
                  read_position++;
                }
          else if (vector_operation == dealii::VectorOperation::max)
            for (const auto &import_range : import_indices_data)
              for (unsigned int j = import_range.first; j < import_range.second;
                   j++)
                {
                  locally_owned_array[j] =
                    internal::get_max(*read_position, locally_owned_array[j]);
                  read_position++;
                }
          else
            for (const auto &import_range : import_indices_data)
              for (unsigned int j = import_range.first; j < import_range.second;
                   j++, read_position++)
                // Below we use relatively large precision in units in the last
                // place (ULP) as this Assert can be easily triggered in
                // p::d::SolutionTransfer. The rationale is that during
                // interpolation on two elements sharing the face, values on
                // this face obtained from each side might be different due to
                // additions being done in different order.
                Assert(*read_position == Number() ||
                         internal::get_abs(locally_owned_array[j] -
                                           *read_position) <=
                           internal::get_abs(locally_owned_array[j] +
                                             *read_position) *
                             100000. *
                             std::numeric_limits<typename numbers::NumberTraits<
                               Number>::real_type>::epsilon(),
                       typename LinearAlgebra::distributed::Vector<
                         Number>::ExcNonMatchingElements(*read_position,
                                                         locally_owned_array[j],
                                                         my_pid));
          std::cout << "CPU values " << std::endl;
          for (const auto value : locally_owned_array)
            std::cout << value << std::endl;
#    else
          if (vector_operation == dealii::VectorOperation::add)
            {
              for (const auto &import_range : import_indices_data)
                {
                  const auto chunk_size =
                    import_range.second - import_range.first;
                  const int n_blocks =
                    1 + (chunk_size - 1) / (::dealii::CUDAWrappers::chunk_size *
                                            ::dealii::CUDAWrappers::block_size);
                  dealii::LinearAlgebra::CUDAWrappers::kernel::vector_bin_op<
                    Number,
                    dealii::LinearAlgebra::CUDAWrappers::kernel::Binop_Addition>
                    <<<n_blocks, dealii::CUDAWrappers::block_size>>>(
                      locally_owned_array.data() + import_range.first,
                      read_position,
                      chunk_size);
                  read_position += chunk_size;
                }
            }
          else
            for (const auto &import_range : import_indices_data)
              {
                const auto chunk_size =
                  import_range.second - import_range.first;
                const cudaError_t cuda_error_code =
                  cudaMemcpy(locally_owned_array.data() + import_range.first,
                             read_position,
                             chunk_size * sizeof(Number),
                             cudaMemcpyDeviceToDevice);
                AssertCuda(cuda_error_code);
                read_position += chunk_size;
              }

          static_assert(
            std::is_same<MemorySpaceType, MemorySpace::CUDA>::value,
            "If we are using the CPU implementation, we should not trigger the restriction");
          Assert(vector_operation == dealii::VectorOperation::insert ||
                   vector_operation == dealii::VectorOperation::add,
                 ExcNotImplemented());
          std::vector<Number> cpu_values(locally_owned_array.size());
          Utilities::CUDA::copy_to_host(locally_owned_array.data(), cpu_values);
          std::cout << "CPU values" << std::endl;
          for (const auto value : cpu_values)
            std::cout << value << std::endl;
#    endif
          AssertDimension(read_position - temporary_storage.data(),
                          n_import_indices());
        }

      {
#    if !(defined(DEAL_II_COMPILER_CUDA_AWARE) && \
          defined(DEAL_II_WITH_CUDA_AWARE_MPI))
        std::cout << "temporary storage" << std::endl;
        for (const auto value : temporary_storage)
          std::cout << value << std::endl;
        std::cout << "owned values" << std::endl;
        for (const auto value : locally_owned_array)
          std::cout << value << std::endl;
        std::cout << "ghost values" << std::endl;
        for (const auto value : ghost_array)
          std::cout << value << std::endl;
#    else
        std::vector<Number> cpu_values_temp(temporary_storage.size());
        Utilities::CUDA::copy_to_host(temporary_storage.data(),
                                      cpu_values_temp);
        std::cout << "temporary storage" << std::endl;
        for (const auto value : cpu_values_temp)
          std::cout << value << std::endl;
        std::vector<Number> cpu_values_owned(locally_owned_array.size());
        Utilities::CUDA::copy_to_host(locally_owned_array.data(),
                                      cpu_values_owned);
        std::cout << "owned values" << std::endl;
        for (const auto value : cpu_values_owned)
          std::cout << value << std::endl;
        std::vector<Number> cpu_values_ghost(ghost_array.size());
        Utilities::CUDA::copy_to_host(ghost_array.data(), cpu_values_ghost);
        std::cout << "ghost values" << std::endl;
        for (const auto value : cpu_values_ghost)
          std::cout << value << std::endl;
#    endif
      }

      // wait for the send operations to complete
      if (requests.size() > 0 && n_ghost_targets > 0)
        {
          const int ierr = MPI_Waitall(n_ghost_targets,
                                       &requests[n_import_targets],
                                       MPI_STATUSES_IGNORE);
          AssertThrowMPI(ierr);
        }
      else
        AssertDimension(n_ghost_indices(), 0);

      // clear the ghost array in case we did not yet do that in the _start
      // function
      if (ghost_array.size() > 0)
        {
          Assert(ghost_array.begin() != nullptr, ExcInternalError());

#    if defined(DEAL_II_COMPILER_CUDA_AWARE) && \
      defined(DEAL_II_WITH_CUDA_AWARE_MPI)
          if (std::is_same<MemorySpaceType, MemorySpace::CUDA>::value)
            {
              Assert(std::is_trivial<Number>::value, ExcNotImplemented());
              cudaMemset(ghost_array.data(),
                         0,
                         sizeof(Number) * n_ghost_indices());
            }
          else
#    endif
            {
#    ifdef DEAL_II_WITH_CXX17
              if constexpr (std::is_trivial<Number>::value)
#    else
            if (std::is_trivial<Number>::value)
#    endif
                {
                  std::memset(ghost_array.data(),
                              0,
                              sizeof(Number) * n_ghost_indices());
                }
              else
                std::fill(ghost_array.data(),
                          ghost_array.data() + n_ghost_indices(),
                          0);
            }
        }

      // clear the compress requests
      requests.resize(0);

      {
#    if !(defined(DEAL_II_COMPILER_CUDA_AWARE) && \
          defined(DEAL_II_WITH_CUDA_AWARE_MPI))
        std::cout << "temporary storage" << std::endl;
        for (const auto value : temporary_storage)
          std::cout << value << std::endl;
        std::cout << "owned values" << std::endl;
        for (const auto value : locally_owned_array)
          std::cout << value << std::endl;
        std::cout << "ghost values" << std::endl;
        for (const auto value : ghost_array)
          std::cout << value << std::endl;
#    else
        std::vector<Number> cpu_values_temp(temporary_storage.size());
        Utilities::CUDA::copy_to_host(temporary_storage.data(),
                                      cpu_values_temp);
        std::cout << "temporary storage" << std::endl;
        for (const auto value : cpu_values_temp)
          std::cout << value << std::endl;
        std::vector<Number> cpu_values_owned(locally_owned_array.size());
        Utilities::CUDA::copy_to_host(locally_owned_array.data(),
                                      cpu_values_owned);
        std::cout << "owned values" << std::endl;
        for (const auto value : cpu_values_owned)
          std::cout << value << std::endl;
        std::vector<Number> cpu_values_ghost(ghost_array.size());
        Utilities::CUDA::copy_to_host(ghost_array.data(), cpu_values_ghost);
        std::cout << "ghost values" << std::endl;
        for (const auto value : cpu_values_ghost)
          std::cout << value << std::endl;
#    endif
      }
    }


#  endif // ifdef DEAL_II_WITH_MPI
#endif   // ifndef DOXYGEN

  } // end of namespace MPI

} // end of namespace Utilities


DEAL_II_NAMESPACE_CLOSE

#endif
