// ---------------------------------------------------------------------
//
// Copyright (C) 2000 - 2017 by the deal.II authors
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

#ifndef dealii_named_selection_h
#define dealii_named_selection_h

#include <deal.II/algorithms/any_data.h>
#include <deal.II/base/config.h>

#include <string>

DEAL_II_NAMESPACE_OPEN

/**
 * Select data from AnyData corresponding to the attached name.
 *
 * Given a list of names to search for (provided by add()), objects of this
 * class provide an index list of the selected data.
 *
 * @author Guido Kanschat, 2009
 */
class NamedSelection
{
public:
  /**
   * Add a new name to be searched for in @p data supplied in initialize().
   *
   * @note Names will be added to the end of the current list.
   */
  void
  add(const std::string &name);

  /**
   * Create the index vector pointing into the AnyData object.
   */
  void
  initialize(const AnyData &data);

  /**
   * The number of names in this object. This function may be used whether
   * initialize() was called before or not.
   */
  unsigned int
  size() const;

  /**
   * Return the corresponding index in the AnyData object supplied to the last
   * initialize(). It is an error if initialize() has not been called before.
   *
   * Indices are in the same order as the calls to add().
   */
  unsigned int
  operator()(unsigned int i) const;

private:
  /**
   * The selected names.
   */
  std::vector<std::string> names;

  /**
   * The index map generated by initialize() and accessed by operator().
   */
  std::vector<unsigned int> indices;
};

inline unsigned int
NamedSelection::size() const
{
  return names.size();
}

inline void
NamedSelection::add(const std::string &s)
{
  names.push_back(s);
}

inline unsigned int
NamedSelection::operator()(unsigned int i) const
{
  Assert(indices.size() == names.size(), ExcNotInitialized());

  AssertIndexRange(i, size());

  return indices[i];
}

DEAL_II_NAMESPACE_CLOSE

#endif
