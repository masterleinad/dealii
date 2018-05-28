// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/base/index_set.h>
#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/mpi.h>

#include <vector>

#ifdef DEAL_II_WITH_TRILINOS
#  ifdef DEAL_II_WITH_MPI
#    include <Epetra_MpiComm.h>
#  endif
#  include <Epetra_Map.h>
#  include <Epetra_SerialComm.h>
#endif

DEAL_II_NAMESPACE_OPEN



#ifdef DEAL_II_WITH_TRILINOS

// the 64-bit path uses a few different names, so put that into a separate
// implementation

#  ifdef DEAL_II_WITH_64BIT_INDICES

IndexSet::IndexSet(const Epetra_Map &map) :
  is_compressed(true),
  index_space_size(1 + map.MaxAllGID64()),
  largest_range(numbers::invalid_unsigned_int)
{
  Assert(map.MinAllGID64() == 0,
         ExcMessage("The Epetra_Map does not contain the global index 0, which "
                    "means some entries are not present on any processor."));

  // For a contiguous map, we do not need to go through the whole data...
  if (map.LinearMap())
    add_range(size_type(map.MinMyGID64()), size_type(map.MaxMyGID64() + 1));
  else
    {
      const size_type n_indices = map.NumMyElements();
      size_type *     indices   = reinterpret_cast<size_type *>(map.MyGlobalElements64());
      add_indices(indices, indices + n_indices);
    }
  compress();
}

#  else

// this is the standard 32-bit implementation

IndexSet::IndexSet(const Epetra_Map &map) :
  is_compressed(true),
  index_space_size(1 + map.MaxAllGID()),
  largest_range(numbers::invalid_unsigned_int)
{
  Assert(map.MinAllGID() == 0,
         ExcMessage("The Epetra_Map does not contain the global index 0, which "
                    "means some entries are not present on any processor."));

  // For a contiguous map, we do not need to go through the whole data...
  if (map.LinearMap())
    add_range(size_type(map.MinMyGID()), size_type(map.MaxMyGID() + 1));
  else
    {
      const size_type n_indices = map.NumMyElements();
      unsigned int *  indices   = reinterpret_cast<unsigned int *>(map.MyGlobalElements());
      add_indices(indices, indices + n_indices);
    }
  compress();
}

#  endif

#endif // ifdef DEAL_II_WITH_TRILINOS



void
IndexSet::add_range(const size_type begin, const size_type end)
{
  Assert((begin < index_space_size) || ((begin == index_space_size) && (end == index_space_size)),
         ExcIndexRangeType<size_type>(begin, 0, index_space_size));
  Assert(end <= index_space_size, ExcIndexRangeType<size_type>(end, 0, index_space_size + 1));
  Assert(begin <= end, ExcIndexRangeType<size_type>(begin, 0, end));

  if (begin != end)
    {
      const Range new_range(begin, end);

      // the new index might be larger than the last index present in the
      // ranges. Then we can skip the binary search
      if (ranges.size() == 0 || begin > ranges.back().end)
        ranges.push_back(new_range);
      else
        ranges.insert(Utilities::lower_bound(ranges.begin(), ranges.end(), new_range), new_range);
      is_compressed = false;
    }
}



void
IndexSet::do_compress() const
{
  // we will, in the following, modify mutable variables. this can only
  // work in multithreaded applications if we lock the data structures
  // via a mutex, so that users can call 'const' functions from threads
  // in parallel (and these 'const' functions can then call compress()
  // which itself calls the current function)
  Threads::Mutex::ScopedLock lock(compress_mutex);

  // see if any of the contiguous ranges can be merged. do not use
  // std::vector::erase in-place as it is quadratic in the number of
  // ranges. since the ranges are sorted by their first index, determining
  // overlap isn't all that hard
  std::vector<Range>::iterator store = ranges.begin();
  for (std::vector<Range>::iterator i = ranges.begin(); i != ranges.end();)
    {
      std::vector<Range>::iterator next = i;
      ++next;

      size_type first_index = i->begin;
      size_type last_index  = i->end;

      // see if we can merge any of the following ranges
      while (next != ranges.end() && (next->begin <= last_index))
        {
          last_index = std::max(last_index, next->end);
          ++next;
        }
      i = next;

      // store the new range in the slot we last occupied
      *store = Range(first_index, last_index);
      ++store;
    }
  // use a compact array with exactly the right amount of storage
  if (store != ranges.end())
    {
      std::vector<Range> new_ranges(ranges.begin(), store);
      ranges.swap(new_ranges);
    }

  // now compute indices within set and the range with most elements
  size_type next_index = 0, largest_range_size = 0;
  for (std::vector<Range>::iterator i = ranges.begin(); i != ranges.end(); ++i)
    {
      Assert(i->begin < i->end, ExcInternalError());

      i->nth_index_in_set = next_index;
      next_index += (i->end - i->begin);
      if (i->end - i->begin > largest_range_size)
        {
          largest_range_size = i->end - i->begin;
          largest_range      = i - ranges.begin();
        }
    }
  is_compressed = true;

  // check that next_index is correct. needs to be after the previous
  // statement because we otherwise will get into an endless loop
  Assert(next_index == n_elements(), ExcInternalError());
}



IndexSet IndexSet::operator&(const IndexSet &is) const
{
  Assert(size() == is.size(), ExcDimensionMismatch(size(), is.size()));

  compress();
  is.compress();

  std::vector<Range>::const_iterator r1 = ranges.begin(), r2 = is.ranges.begin();
  IndexSet                           result(size());

  while ((r1 != ranges.end()) && (r2 != is.ranges.end()))
    {
      // if r1 and r2 do not overlap at all, then move the pointer that sits
      // to the left of the other up by one
      if (r1->end <= r2->begin)
        ++r1;
      else if (r2->end <= r1->begin)
        ++r2;
      else
        {
          // the ranges must overlap somehow
          Assert(((r1->begin <= r2->begin) && (r1->end > r2->begin)) ||
                   ((r2->begin <= r1->begin) && (r2->end > r1->begin)),
                 ExcInternalError());

          // add the overlapping range to the result
          result.add_range(std::max(r1->begin, r2->begin), std::min(r1->end, r2->end));

          // now move that iterator that ends earlier one up. note that it has
          // to be this one because a subsequent range may still have a chance
          // of overlapping with the range that ends later
          if (r1->end <= r2->end)
            ++r1;
          else
            ++r2;
        }
    }

  result.compress();
  return result;
}



IndexSet
IndexSet::get_view(const size_type begin, const size_type end) const
{
  Assert(begin <= end, ExcMessage("End index needs to be larger or equal to begin index!"));
  Assert(end <= size(), ExcMessage("Given range exceeds index set dimension"));

  IndexSet                           result(end - begin);
  std::vector<Range>::const_iterator r1 = ranges.begin();

  while (r1 != ranges.end())
    {
      if ((r1->end > begin) && (r1->begin < end))
        {
          result.add_range(std::max(r1->begin, begin) - begin, std::min(r1->end, end) - begin);
        }
      else if (r1->begin >= end)
        break;

      ++r1;
    }

  result.compress();
  return result;
}



void
IndexSet::subtract_set(const IndexSet &other)
{
  compress();
  other.compress();
  is_compressed = false;


  // we save new ranges to be added to our IndexSet in an temporary vector and
  // add all of them in one go at the end.
  std::vector<Range> new_ranges;

  std::vector<Range>::iterator own_it   = ranges.begin();
  std::vector<Range>::iterator other_it = other.ranges.begin();

  while (own_it != ranges.end() && other_it != other.ranges.end())
    {
      // advance own iterator until we get an overlap
      if (own_it->end <= other_it->begin)
        {
          ++own_it;
          continue;
        }
      // we are done with other_it, so advance
      if (own_it->begin >= other_it->end)
        {
          ++other_it;
          continue;
        }

      // Now own_it and other_it overlap.  First save the part of own_it that
      // is before other_it (if not empty).
      if (own_it->begin < other_it->begin)
        {
          Range r(own_it->begin, other_it->begin);
          r.nth_index_in_set = 0; // fix warning of unused variable
          new_ranges.push_back(r);
        }
      // change own_it to the sub range behind other_it. Do not delete own_it
      // in any case. As removal would invalidate iterators, we just shrink
      // the range to an empty one.
      own_it->begin = other_it->end;
      if (own_it->begin > own_it->end)
        {
          own_it->begin = own_it->end;
          ++own_it;
        }

      // continue without advancing iterators, the right one will be advanced
      // next.
    }

  // Now delete all empty ranges we might
  // have created.
  for (std::vector<Range>::iterator it = ranges.begin(); it != ranges.end();)
    {
      if (it->begin >= it->end)
        it = ranges.erase(it);
      else
        ++it;
    }

  // done, now add the temporary ranges
  const std::vector<Range>::iterator end = new_ranges.end();
  for (std::vector<Range>::iterator it = new_ranges.begin(); it != end; ++it)
    add_range(it->begin, it->end);

  compress();
}



IndexSet::size_type
IndexSet::pop_back()
{
  Assert(is_empty() == false,
         ExcMessage("pop_back() failed, because this IndexSet contains no entries."));

  const size_type index = ranges.back().end - 1;
  --ranges.back().end;

  if (ranges.back().begin == ranges.back().end)
    ranges.pop_back();

  return index;
}



IndexSet::size_type
IndexSet::pop_front()
{
  Assert(is_empty() == false,
         ExcMessage("pop_front() failed, because this IndexSet contains no entries."));

  const size_type index = ranges.front().begin;
  ++ranges.front().begin;

  if (ranges.front().begin == ranges.front().end)
    ranges.erase(ranges.begin());

  // We have to set this in any case, because nth_index_in_set is no longer
  // up to date for all but the first range
  is_compressed = false;

  return index;
}



void
IndexSet::add_indices(const IndexSet &other, const unsigned int offset)
{
  if ((this == &other) && (offset == 0))
    return;

  Assert(other.ranges.size() == 0 || other.ranges.back().end - 1 < index_space_size,
         ExcIndexRangeType<size_type>(other.ranges.back().end - 1, 0, index_space_size));

  compress();
  other.compress();

  std::vector<Range>::const_iterator r1 = ranges.begin(), r2 = other.ranges.begin();

  std::vector<Range> new_ranges;
  // just get the start and end of the ranges right in this method, everything
  // else will be done in compress()
  while (r1 != ranges.end() || r2 != other.ranges.end())
    {
      // the two ranges do not overlap or we are at the end of one of the
      // ranges
      if (r2 == other.ranges.end() || (r1 != ranges.end() && r1->end < (r2->begin + offset)))
        {
          new_ranges.push_back(*r1);
          ++r1;
        }
      else if (r1 == ranges.end() || (r2->end + offset) < r1->begin)
        {
          new_ranges.emplace_back(r2->begin + offset, r2->end + offset);
          ++r2;
        }
      else
        {
          // ok, we do overlap, so just take the combination of the current
          // range (do not bother to merge with subsequent ranges)
          Range next(std::min(r1->begin, r2->begin + offset), std::max(r1->end, r2->end + offset));
          new_ranges.push_back(next);
          ++r1;
          ++r2;
        }
    }
  ranges.swap(new_ranges);

  is_compressed = false;
  compress();
}



void
IndexSet::write(std::ostream &out) const
{
  compress();
  out << size() << " ";
  out << ranges.size() << std::endl;
  std::vector<Range>::const_iterator r = ranges.begin();
  for (; r != ranges.end(); ++r)
    {
      out << r->begin << " " << r->end << std::endl;
    }
}



void
IndexSet::read(std::istream &in)
{
  AssertThrow(in, ExcIO());

  size_type    s;
  unsigned int n_ranges;

  in >> s >> n_ranges;
  ranges.clear();
  set_size(s);
  for (unsigned int i = 0; i < n_ranges; ++i)
    {
      AssertThrow(in, ExcIO());

      size_type b, e;
      in >> b >> e;
      add_range(b, e);
    }
}


void
IndexSet::block_write(std::ostream &out) const
{
  AssertThrow(out, ExcIO());
  out.write(reinterpret_cast<const char *>(&index_space_size), sizeof(index_space_size));
  size_t n_ranges = ranges.size();
  out.write(reinterpret_cast<const char *>(&n_ranges), sizeof(n_ranges));
  if (ranges.empty() == false)
    out.write(reinterpret_cast<const char *>(&*ranges.begin()), ranges.size() * sizeof(Range));
  AssertThrow(out, ExcIO());
}

void
IndexSet::block_read(std::istream &in)
{
  size_type size;
  size_t    n_ranges;
  in.read(reinterpret_cast<char *>(&size), sizeof(size));
  in.read(reinterpret_cast<char *>(&n_ranges), sizeof(n_ranges));
  // we have to clear ranges first
  ranges.clear();
  set_size(size);
  ranges.resize(n_ranges, Range(0, 0));
  if (n_ranges)
    in.read(reinterpret_cast<char *>(&*ranges.begin()), ranges.size() * sizeof(Range));

  do_compress(); // needed so that largest_range can be recomputed
}



void
IndexSet::fill_index_vector(std::vector<size_type> &indices) const
{
  compress();

  indices.clear();
  indices.reserve(n_elements());

  for (std::vector<Range>::iterator it = ranges.begin(); it != ranges.end(); ++it)
    for (size_type i = it->begin; i < it->end; ++i)
      indices.push_back(i);

  Assert(indices.size() == n_elements(), ExcInternalError());
}



#ifdef DEAL_II_WITH_TRILINOS

Epetra_Map
IndexSet::make_trilinos_map(const MPI_Comm &communicator, const bool overlapping) const
{
  compress();
  (void)communicator;

#  ifdef DEBUG
  if (!overlapping)
    {
      const size_type n_global_elements = Utilities::MPI::sum(n_elements(), communicator);
      Assert(n_global_elements == size(),
             ExcMessage("You are trying to create an Epetra_Map object "
                        "that partitions elements of an index set "
                        "between processors. However, the union of the "
                        "index sets on different processors does not "
                        "contain all indices exactly once: the sum of "
                        "the number of entries the various processors "
                        "want to store locally is " +
                        Utilities::to_string(n_global_elements) +
                        " whereas the total size of the object to be "
                        "allocated is " +
                        Utilities::to_string(size()) +
                        ". In other words, there are "
                        "either indices that are not spoken for "
                        "by any processor, or there are indices that are "
                        "claimed by multiple processors."));
    }
#  endif

  // Find out if the IndexSet is ascending and 1:1. This corresponds to a
  // linear EpetraMap. Overlapping IndexSets are never 1:1.
  const bool linear = overlapping ? false : is_ascending_and_one_to_one(communicator);

  if (linear)
    return Epetra_Map(TrilinosWrappers::types::int_type(size()),
                      TrilinosWrappers::types::int_type(n_elements()),
                      0,
#  ifdef DEAL_II_WITH_MPI
                      Epetra_MpiComm(communicator)
#  else
                      Epetra_SerialComm()
#  endif
    );
  else
    {
      std::vector<size_type> indices;
      fill_index_vector(indices);
      return Epetra_Map(
        TrilinosWrappers::types::int_type(-1),
        TrilinosWrappers::types::int_type(n_elements()),
        (n_elements() > 0 ? reinterpret_cast<TrilinosWrappers::types::int_type *>(indices.data()) :
                            nullptr),
        0,
#  ifdef DEAL_II_WITH_MPI
        Epetra_MpiComm(communicator)
#  else
        Epetra_SerialComm()
#  endif
      );
    }
}
#endif



bool
IndexSet::is_ascending_and_one_to_one(const MPI_Comm &communicator) const
{
  // If the sum of local elements does not add up to the total size,
  // the IndexSet can't be complete.
  const size_type n_global_elements = Utilities::MPI::sum(n_elements(), communicator);
  if (n_global_elements != size())
    return false;

#ifdef DEAL_II_WITH_MPI
  // Non-contiguous IndexSets can't be linear.
  const bool all_contiguous = (Utilities::MPI::min(is_contiguous() ? 1 : 0, communicator) == 1);
  if (!all_contiguous)
    return false;

  bool is_globally_ascending = true;
  // we know that there is only one interval
  types::global_dof_index first_local_dof =
    (n_elements() > 0) ? *(begin_intervals()->begin()) : numbers::invalid_dof_index;

  const unsigned int my_rank = Utilities::MPI::this_mpi_process(communicator);
  const std::vector<types::global_dof_index> global_dofs =
    Utilities::MPI::gather(communicator, first_local_dof, 0);

  if (my_rank == 0)
    {
      // find out if the received std::vector is ascending
      types::global_dof_index index = 0;
      while (global_dofs[index] == numbers::invalid_dof_index)
        ++index;
      types::global_dof_index old_dof = global_dofs[index++];
      for (; index < global_dofs.size(); ++index)
        {
          const types::global_dof_index new_dof = global_dofs[index];
          if (new_dof != numbers::invalid_dof_index)
            {
              if (new_dof <= old_dof)
                {
                  is_globally_ascending = false;
                  break;
                }
              else
                old_dof = new_dof;
            }
        }
    }

  // now broadcast the result
  int is_ascending = is_globally_ascending ? 1 : 0;
  int ierr         = MPI_Bcast(&is_ascending, 1, MPI_INT, 0, communicator);
  AssertThrowMPI(ierr);

  return (is_ascending == 1);
#else
  return true;
#endif // DEAL_II_WITH_MPI
}



std::size_t
IndexSet::memory_consumption() const
{
  return (MemoryConsumption::memory_consumption(ranges) +
          MemoryConsumption::memory_consumption(is_compressed) +
          MemoryConsumption::memory_consumption(index_space_size) + sizeof(compress_mutex));
}



DEAL_II_NAMESPACE_CLOSE
