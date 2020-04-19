// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2019 by the deal.II authors
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


#ifndef dealii_vector_tools_templates_h
#define dealii_vector_tools_templates_h

#include <deal.II/base/config.h>

#include <deal.II/base/derivative_form.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/polynomials_piecewise.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_nedelec_sz.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_dg0.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/intergrid_map.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_epetra_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_tpetra_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/range/iterator_range.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <list>
#include <numeric>
#include <set>
#include <typeinfo>
#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace VectorTools
{
  namespace internal
  {} // namespace internal



  namespace internal
  {
    /**
     * Return whether the cell and all of its descendants are locally owned.
     */
    template <typename cell_iterator>
    bool
    is_locally_owned(const cell_iterator &cell)
    {
      if (cell->is_active())
        return cell->is_locally_owned();

      for (unsigned int c = 0; c < cell->n_children(); ++c)
        if (!is_locally_owned(cell->child(c)))
          return false;

      return true;
    }
  } // namespace internal


  namespace internal
  {}



  namespace internal
  {} // namespace internal



  namespace internals
  {} // namespace internals



  namespace internal
  {}



} // namespace VectorTools
} // namespace VectorTools

DEAL_II_NAMESPACE_CLOSE

#endif
