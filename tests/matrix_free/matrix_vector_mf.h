// ---------------------------------------------------------------------
//
// Copyright (C) 2013 - 2017 by the deal.II authors
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


// this is a template for matrix-vector products with the Helmholtz equation
// (zero and first derivatives) on different kinds of meshes (Cartesian,
// general, with and without hanging nodes). It also tests the multithreading
// in case it was enabled

#include "../tests.h"

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>


template <int dim, int fe_degree, typename VectorType, int n_q_points_1d>
void
helmholtz_operator(const MatrixFree<dim, typename VectorType::value_type>& data,
                   VectorType&                                             dst,
                   const VectorType&                                       src,
                   const std::pair<unsigned int, unsigned int>& cell_range)
{
  typedef typename VectorType::value_type                Number;
  FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number> fe_eval(data);
  const unsigned int n_q_points = fe_eval.n_q_points;

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, true, false);
      for(unsigned int q = 0; q < n_q_points; ++q)
        {
          fe_eval.submit_value(Number(10) * fe_eval.get_value(q), q);
          fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
        }
      fe_eval.integrate(true, true);
      fe_eval.distribute_local_to_global(dst);
    }
}



template <int dim,
          int fe_degree,
          typename Number,
          typename VectorType = Vector<Number>,
          int n_q_points_1d   = fe_degree + 1>
class MatrixFreeTest
{
public:
  typedef VectorizedArray<Number> vector_t;

  MatrixFreeTest(const MatrixFree<dim, Number>& data_in) : data(data_in){};

  void
  vmult(VectorType& dst, const VectorType& src) const
  {
    dst = 0;
    const std::function<void(
      const MatrixFree<dim, typename VectorType::value_type>&,
      VectorType&,
      const VectorType&,
      const std::pair<unsigned int, unsigned int>&)>
      wrap = helmholtz_operator<dim, fe_degree, VectorType, n_q_points_1d>;
    data.cell_loop(wrap, dst, src);
  };

private:
  const MatrixFree<dim, Number>& data;
};
