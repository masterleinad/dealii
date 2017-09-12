// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2016 by the deal.II authors
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

#include <deal.II/base/config.h>

// Required for instantiation of template functions
#ifdef DEAL_II_WITH_TRILINOS

DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#  include <Sacado.hpp>
DEAL_II_ENABLE_EXTRA_DIAGNOSTICS

#include <deal.II/differentiation/ad/sacado_product_types.h>
#endif

#include <deal.II/base/symmetric_tensor.templates.h>

DEAL_II_NAMESPACE_OPEN


// provide definitions for static members
template <int rank, int dim, typename Number>
const unsigned int SymmetricTensor<rank,dim,Number>::dimension;

template <int rank, int dim, typename Number>
const unsigned int SymmetricTensor<rank,dim,Number>::n_independent_components;


// explicit instantiations
#include "symmetric_tensor.inst"


DEAL_II_NAMESPACE_CLOSE
