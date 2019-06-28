// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2019 by the deal.II authors
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

#include <deal.II/base/vectorization.h>

DEAL_II_NAMESPACE_OPEN

#if DEAL_II_COMPILER_VECTORIZATION_LEVEL >= 1 && !defined(DEAL_II_MSVC)
constexpr unsigned int VectorizedArray<double>::n_array_elements;
constexpr unsigned int VectorizedArray<float>::n_array_elements;
static_assert(
  std::is_standard_layout<VectorizedArray<double>>::value,
  "VectorizedArray<double> must has standard (C compatible) layout");
static_assert(std::is_standard_layout<VectorizedArray<float>>::value,
              "VectorizedArray<float> must has standard (C compatible) layout");
#endif

DEAL_II_NAMESPACE_CLOSE
