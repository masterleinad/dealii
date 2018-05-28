// ---------------------------------------------------------------------
//
// Copyright (C) 2008 - 2017 by the deal.II authors
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

#include <deal.II/base/memory_consumption.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

DEAL_II_NAMESPACE_OPEN



template <typename ForwardIterator>
void
DynamicSparsityPattern::Line::add_entries(ForwardIterator begin,
                                          ForwardIterator end,
                                          const bool      indices_are_sorted)
{
  const int n_elements = end - begin;
  if (n_elements <= 0)
    return;

  const size_type stop_size = entries.size() + n_elements;

  if (indices_are_sorted == true && n_elements > 3)
    {
      // in debug mode, check whether the
      // indices really are sorted.
#ifdef DEBUG
      {
        ForwardIterator test = begin, test1 = begin;
        ++test1;
        for (; test1 != end; ++test, ++test1)
          Assert(*test1 > *test, ExcInternalError());
      }
#endif

      if (entries.size() == 0 || entries.back() < *begin)
        {
          entries.insert(entries.end(), begin, end);
          return;
        }

      // find a possible insertion point for
      // the first entry. check whether the
      // first entry is a duplicate before
      // actually doing something.
      ForwardIterator                  my_it = begin;
      size_type                        col   = *my_it;
      std::vector<size_type>::iterator it =
        Utilities::lower_bound(entries.begin(), entries.end(), col);
      while (*it == col)
        {
          ++my_it;
          if (my_it == end)
            break;
          col = *my_it;
          // check the very next entry in the
          // current array
          ++it;
          if (it == entries.end())
            break;
          if (*it > col)
            break;
          if (*it == col)
            continue;
          // ok, it wasn't the very next one, do a
          // binary search to find the insert point
          it = Utilities::lower_bound(it, entries.end(), col);
          if (it == entries.end())
            break;
        }
      // all input entries were duplicates.
      if (my_it == end)
        return;

      // resize vector by just inserting the
      // list
      const size_type pos1 = it - entries.begin();
      Assert(pos1 <= entries.size(), ExcInternalError());
      entries.insert(it, my_it, end);
      it = entries.begin() + pos1;
      Assert(entries.size() >= (size_type)(it - entries.begin()), ExcInternalError());

      // now merge the two lists.
      std::vector<size_type>::iterator it2 = it + (end - my_it);

      // as long as there are indices both in
      // the end of the entries list and in the
      // input list
      while (my_it != end && it2 != entries.end())
        {
          if (*my_it < *it2)
            *it++ = *my_it++;
          else if (*my_it == *it2)
            {
              *it++ = *it2++;
              ++my_it;
            }
          else
            *it++ = *it2++;
        }
      // in case there are indices left in the
      // input list
      while (my_it != end)
        *it++ = *my_it++;

      // in case there are indices left in the
      // end of entries
      while (it2 != entries.end())
        *it++ = *it2++;

      // resize and return
      const size_type new_size = it - entries.begin();
      Assert(new_size <= stop_size, ExcInternalError());
      entries.resize(new_size);
      return;
    }

  // unsorted case or case with too few
  // elements
  ForwardIterator my_it = begin;

  // If necessary, increase the size of the
  // array.
  if (stop_size > entries.capacity())
    entries.reserve(stop_size);

  size_type                        col = *my_it;
  std::vector<size_type>::iterator it, it2;
  // insert the first element as for one
  // entry only first check the last
  // element (or if line is still empty)
  if ((entries.size() == 0) || (entries.back() < col))
    {
      entries.push_back(col);
      it = entries.end() - 1;
    }
  else
    {
      // do a binary search to find the place
      // where to insert:
      it2 = Utilities::lower_bound(entries.begin(), entries.end(), col);

      // If this entry is a duplicate, continue
      // immediately Insert at the right place
      // in the vector. Vector grows
      // automatically to fit elements. Always
      // doubles its size.
      if (*it2 != col)
        it = entries.insert(it2, col);
      else
        it = it2;
    }

  ++my_it;
  // Now try to be smart and insert with
  // bias in the direction we are
  // walking. This has the advantage that
  // for sorted lists, we always search in
  // the right direction, what should
  // decrease the work needed in here.
  for (; my_it != end; ++my_it)
    {
      col = *my_it;
      // need a special insertion command when
      // we're at the end of the list
      if (col > entries.back())
        {
          entries.push_back(col);
          it = entries.end() - 1;
        }
      // search to the right (preferred search
      // direction)
      else if (col > *it)
        {
          it2 = Utilities::lower_bound(it++, entries.end(), col);
          if (*it2 != col)
            it = entries.insert(it2, col);
        }
      // search to the left
      else if (col < *it)
        {
          it2 = Utilities::lower_bound(entries.begin(), it, col);
          if (*it2 != col)
            it = entries.insert(it2, col);
        }
      // if we're neither larger nor smaller,
      // then this was a duplicate and we can
      // just continue.
    }
}


DynamicSparsityPattern::size_type
DynamicSparsityPattern::Line::memory_consumption() const
{
  return entries.capacity() * sizeof(size_type) + sizeof(Line);
}


DynamicSparsityPattern::DynamicSparsityPattern() : have_entries(false), rows(0), cols(0), rowset(0)
{}



DynamicSparsityPattern::DynamicSparsityPattern(const DynamicSparsityPattern &s) :
  Subscriptor(),
  have_entries(false),
  rows(0),
  cols(0),
  rowset(0)
{
  (void)s;
  Assert(s.rows == 0 && s.cols == 0,
         ExcMessage("This constructor can only be called if the provided argument "
                    "is the sparsity pattern for an empty matrix. This constructor can "
                    "not be used to copy-construct a non-empty sparsity pattern."));
}



DynamicSparsityPattern::DynamicSparsityPattern(const size_type m,
                                               const size_type n,
                                               const IndexSet &rowset_) :
  have_entries(false),
  rows(0),
  cols(0),
  rowset(0)
{
  reinit(m, n, rowset_);
}


DynamicSparsityPattern::DynamicSparsityPattern(const IndexSet &rowset_) :
  have_entries(false),
  rows(0),
  cols(0),
  rowset(0)
{
  reinit(rowset_.size(), rowset_.size(), rowset_);
}


DynamicSparsityPattern::DynamicSparsityPattern(const size_type n) :
  have_entries(false),
  rows(0),
  cols(0),
  rowset(0)
{
  reinit(n, n);
}



DynamicSparsityPattern &
DynamicSparsityPattern::operator=(const DynamicSparsityPattern &s)
{
  (void)s;
  Assert(s.rows == 0 && s.cols == 0,
         ExcMessage("This operator can only be called if the provided argument "
                    "is the sparsity pattern for an empty matrix. This operator can "
                    "not be used to copy a non-empty sparsity pattern."));

  Assert(rows == 0 && cols == 0,
         ExcMessage("This operator can only be called if the current object is"
                    "empty."));

  return *this;
}



void
DynamicSparsityPattern::reinit(const size_type m, const size_type n, const IndexSet &rowset_)
{
  have_entries = false;
  rows         = m;
  cols         = n;
  rowset       = rowset_;

  Assert(rowset.size() == 0 || rowset.size() == m,
         ExcMessage("The IndexSet argument to this function needs to either "
                    "be empty (indicating the complete set of rows), or have size "
                    "equal to the desired number of rows as specified by the "
                    "first argument to this function. (Of course, the number "
                    "of indices in this IndexSet may be less than the number "
                    "of rows, but the *size* of the IndexSet must be equal.)"));

  std::vector<Line> new_lines(rowset.size() == 0 ? rows : rowset.n_elements());
  lines.swap(new_lines);
}



void
DynamicSparsityPattern::compress()
{}



bool
DynamicSparsityPattern::empty() const
{
  return ((rows == 0) && (cols == 0));
}



DynamicSparsityPattern::size_type
DynamicSparsityPattern::max_entries_per_row() const
{
  if (!have_entries)
    return 0;

  size_type m = 0;
  for (size_type i = 0; i < lines.size(); ++i)
    {
      m = std::max(m, static_cast<size_type>(lines[i].entries.size()));
    }

  return m;
}



bool
DynamicSparsityPattern::exists(const size_type i, const size_type j) const
{
  Assert(i < rows, ExcIndexRange(i, 0, rows));
  Assert(j < cols, ExcIndexRange(j, 0, cols));
  Assert(rowset.size() == 0 || rowset.is_element(i), ExcInternalError());

  if (!have_entries)
    return false;

  const size_type rowindex = rowset.size() == 0 ? i : rowset.index_within_set(i);

  return std::binary_search(lines[rowindex].entries.begin(), lines[rowindex].entries.end(), j);
}



void
DynamicSparsityPattern::symmetrize()
{
  Assert(rows == cols, ExcNotQuadratic());

  // loop over all elements presently
  // in the sparsity pattern and add
  // the transpose element. note:
  //
  // 1. that the sparsity pattern
  // changes which we work on, but
  // not the present row
  //
  // 2. that the @p{add} function can
  // be called on elements that
  // already exist without any harm
  for (size_type row = 0; row < lines.size(); ++row)
    {
      const size_type rowindex = rowset.size() == 0 ? row : rowset.nth_index_in_set(row);

      for (std::vector<size_type>::const_iterator j = lines[row].entries.begin();
           j != lines[row].entries.end();
           ++j)
        // add the transpose entry if
        // this is not the diagonal
        if (rowindex != *j)
          add(*j, rowindex);
    }
}



template <typename SparsityPatternTypeLeft, typename SparsityPatternTypeRight>
void
DynamicSparsityPattern::compute_mmult_pattern(const SparsityPatternTypeLeft & left,
                                              const SparsityPatternTypeRight &right)
{
  Assert(left.n_cols() == right.n_rows(), ExcDimensionMismatch(left.n_cols(), right.n_rows()));

  this->reinit(left.n_rows(), right.n_cols());

  typename SparsityPatternTypeLeft::iterator it_left = left.begin(), end_left = left.end();
  for (; it_left != end_left; ++it_left)
    {
      const unsigned int j = it_left->column();

      // We are sitting on entry (i,j) of the left sparsity pattern. We then
      // need to add all entries (i,k) to the final sparsity pattern where (j,k)
      // exists in the right sparsity pattern -- i.e., we need to iterate over
      // row j.
      typename SparsityPatternTypeRight::iterator it_right  = right.begin(j),
                                                  end_right = right.end(j);
      for (; it_right != end_right; ++it_right)
        this->add(it_left->row(), it_right->column());
    }
}



void
DynamicSparsityPattern::print(std::ostream &out) const
{
  for (size_type row = 0; row < lines.size(); ++row)
    {
      out << '[' << (rowset.size() == 0 ? row : rowset.nth_index_in_set(row));

      for (std::vector<size_type>::const_iterator j = lines[row].entries.begin();
           j != lines[row].entries.end();
           ++j)
        out << ',' << *j;

      out << ']' << std::endl;
    }

  AssertThrow(out, ExcIO());
}



void
DynamicSparsityPattern::print_gnuplot(std::ostream &out) const
{
  for (size_type row = 0; row < lines.size(); ++row)
    {
      const size_type rowindex = rowset.size() == 0 ? row : rowset.nth_index_in_set(row);

      for (std::vector<size_type>::const_iterator j = lines[row].entries.begin();
           j != lines[row].entries.end();
           ++j)
        // while matrix entries are usually
        // written (i,j), with i vertical and
        // j horizontal, gnuplot output is
        // x-y, that is we have to exchange
        // the order of output
        out << *j << " " << -static_cast<signed int>(rowindex) << std::endl;
    }


  AssertThrow(out, ExcIO());
}



DynamicSparsityPattern::size_type
DynamicSparsityPattern::bandwidth() const
{
  size_type b = 0;
  for (size_type row = 0; row < lines.size(); ++row)
    {
      const size_type rowindex = rowset.size() == 0 ? row : rowset.nth_index_in_set(row);

      for (std::vector<size_type>::const_iterator j = lines[row].entries.begin();
           j != lines[row].entries.end();
           ++j)
        if (static_cast<size_type>(std::abs(static_cast<int>(rowindex - *j))) > b)
          b = std::abs(static_cast<signed int>(rowindex - *j));
    }

  return b;
}



DynamicSparsityPattern::size_type
DynamicSparsityPattern::n_nonzero_elements() const
{
  if (!have_entries)
    return 0;

  size_type n = 0;
  for (size_type i = 0; i < lines.size(); ++i)
    {
      n += lines[i].entries.size();
    }

  return n;
}


DynamicSparsityPattern::size_type
DynamicSparsityPattern::memory_consumption() const
{
  size_type mem =
    sizeof(DynamicSparsityPattern) + MemoryConsumption::memory_consumption(rowset) - sizeof(rowset);

  for (size_type i = 0; i < lines.size(); ++i)
    mem += MemoryConsumption::memory_consumption(lines[i]);

  return mem;
}


// explicit instantiations
template void
DynamicSparsityPattern::Line::add_entries(size_type *, size_type *, const bool);
template void
DynamicSparsityPattern::Line::add_entries(const size_type *, const size_type *, const bool);
#ifndef DEAL_II_VECTOR_ITERATOR_IS_POINTER
template void
DynamicSparsityPattern::Line::add_entries(std::vector<size_type>::iterator,
                                          std::vector<size_type>::iterator,
                                          const bool);
#endif

template void
DynamicSparsityPattern::compute_mmult_pattern(const DynamicSparsityPattern &,
                                              const DynamicSparsityPattern &);
template void
DynamicSparsityPattern::compute_mmult_pattern(const DynamicSparsityPattern &,
                                              const SparsityPattern &);
template void
DynamicSparsityPattern::compute_mmult_pattern(const SparsityPattern &,
                                              const DynamicSparsityPattern &);
template void
DynamicSparsityPattern::compute_mmult_pattern(const SparsityPattern &, const SparsityPattern &);

DEAL_II_NAMESPACE_CLOSE
