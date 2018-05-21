// ---------------------------------------------------------------------
//
// Copyright (C) 2009 - 2014 by the deal.II authors
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

#ifndef dealii_std_cxx11_condition_variable_h
#define dealii_std_cxx11_condition_variable_h

#include <deal.II/base/config.h>

#include <condition_variable>
DEAL_II_NAMESPACE_OPEN
namespace std_cxx11
{
  using std::adopt_lock;
  using std::condition_variable;
  using std::unique_lock;
} // namespace std_cxx11
DEAL_II_NAMESPACE_CLOSE

// then allow using the old namespace name instead of the new one
DEAL_II_NAMESPACE_OPEN
namespace std_cxx1x= std_cxx11;
DEAL_II_NAMESPACE_CLOSE

#endif
