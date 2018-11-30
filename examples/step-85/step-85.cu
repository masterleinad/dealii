/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2018 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Bruno Turcksin, Oak Ridge National Laboratory
 */

// First include the necessary files from the deal.II libary.
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>

// This includes the data structures for the implementation of matrix-free
// methods on GPU
#include <deal.II/matrix_free/cuda_matrix_free.h>
#include <deal.II/matrix_free/cuda_fe_evaluation.h>

namespace Step85
{
  using namespace dealii;

  //  template <int dim>
  //  class Coefficient : public Function<dim>
  //  {
  //  public:
  //    Coefficient()
  //      : Function<dim>()
  //    {}
  //
  //    virtual double value(const Point<dim> & p,
  //                         const unsigned int component = 0) const override;
  //  };
  //
  //
  //
  //  template <int dim>
  //  double Coefficient<dim>::value(const Point<dim> &p,
  //                                 const unsigned int /*component*/)
  //  {
  //    return 1. / (0.05 + 2. * p.square());
  //  }


  template <int dim, int fe_degree>
  class LaplaceOperatorQuad
  {
  public:
    __device__ void operator()(CUDA::FEEvaluation<dim, fe_degree> *fe_eval,
                               const unsigned int q_point) const;
  };



  template <int dim, int fe_degree>
  class LocalLaplaceOperator
  {
  public:
    __device__ void operator()(
      const unsigned int                                          cell,
      const typename CUDAWrappers::MatrixFree<dim, Number>::Data *gpu_data,
      CUDAWrappers::SharedData<dim, Number> *                     shared_data,
      const Number *                                              src,
      Number *                                                    dst) const;
  };

  template <int dim, int fe_degree>
  __device__ void LocalLaplaceOperator<dim, fe_free>::operator()(
    const unsigned int                                          cell,
    const typename CUDAWrappers::MatrixFree<dim, Number>::Data *gpu_data,
    CUDAWrappers::SharedData<dim, Number> *                     shared_data,
    const Number *                                              src,
    Number *                                                    dst) const
  {
    CUDAWrappers::FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number>
      fe_eval(cell, gpu_data, shared_data);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(false true);
    fe_eval.apply_quad_point_operations(LaplaceOperatorQuad<dim, fe_degree>());
    fe_eval.integrate(false, true);
    fe_eval.distribute_local_to_global(dst);
  }

  template <int dim, int fe_degree>
  class LaplaceOperator
  {
  public:
    LaplaceOperator();

    void evaluate_coefficient(const Coefficient<dim> &coefficient_function);

    void compute_inverse_diagonal();

  private:
    void
    vmult(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &data,
          const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>
            &src) const;
  };
} // namespace Step85
