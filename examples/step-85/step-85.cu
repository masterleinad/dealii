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

  template <int dim, int fe_degree>
  class HelmholtzOperatorQuad
  {
  public:
    __device__ void operator()(CUDA::FEEvaluation<dim, fe_degree> *fe_eval,
                               const unsigned int q_point) const;
  };



  template <int dim, int fe_degree>
  __device__ void HelmholtzOperatorQuad::
                  operator()(CUDA::FEEvaluation<dim, fe_degree> *fe_eval,
             const unsigned int                  q_point) const
  {
    fe_eval->submit_value(10. * fe_eval->get_value(q), q);
    fe_eval->submit_gradient(fe_eval->get_gradient(q), q);
  }



  template <int dim, int fe_degree>
  class LocalHelmholtzOperator
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
  __device__ void LocalHelmholtzOperator<dim, fe_free>::operator()(
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
    fe_eval.apply_quad_point_operations(
      HelmholtzOperatorQuad<dim, fe_degree>());
    fe_eval.integrate(false, true);
    fe_eval.distribute_local_to_global(dst);
  }



  template <int dim, int fe_degree>
  class HelmholtzOperator
  {
  public:
    HelmholtzOperator(const DoFHandler<dim> &          dof_handler,
                      const AffineConstraints<double> &constraints);

    // TODO add varying coefficient using a lambda function
    // void evaluate_coefficient(const Coefficient<dim> &coefficient_function);

    //    void compute_inverse_diagonal();

    void
    vmult(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>
            &src) const;

  private:
    CUDAWrappers::MatrixFree<dim, double> mf_data;
  };



  template <int dim, int fe_degree>
  HelmholtzOperator<dim, fe_degree>::HelmholtzOperator(
    const DoFHandler<dim> &          dof_handler,
    const AffineConstraints<double> &constraints)
  {
    MappingQGeneric<dim> mapping(fe_degree);
    typename CUDAWrappers::MatrixFree<dim, Number>::AdditionalData
      additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients |
                                           update_JxW_values |
                                           update_quadrature_points;
    const QGauss<1> quad(fe_degree + 1);
    mf_data.reinit(mapping, dof_handler, constraints, quad, additional_data);
  }



  template <int dim, int fe_degree>
  void HelmholtzOperator<dim, fe_degre>::vmult(
    LinearAlgebra::distributed::VectorVector<double, MemorySpace::CUDA> &dst,
    const LinearAlgebra::distributed::VectorVector<double, MemorySpace::CUDA>
      &src) const
  {
    dst = static_cast<Number>(0.);
    LocalHelmholtzOperator<dim, fe_degree> helmholtz_operator();
    data.cell_loop(helmholtz_operator, src, dst);
    data.copy_constrained_values(src, dst);
  }



  template <int dim, int fe_degree>
  class HelmholtzProblem
  {
  public:
    HelmholtzProblem();
    ~HelmholtzProblem();

    void run();

  private:
    void setup_system();
    // TODO just do it on the host and then move to the GPU
    // void assemble_rhs();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;

    MPI_Comm mpi_communicator;

    parallel::distributed::Triangulation<dim> triangulation;

    DoFHandler<dim> dof_handler;
    FE_Q<dim>       fe;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double>                          constraints;
    std::unique_ptr<HelmholtzOperator<dim, fe_degree>> system_matrix;

    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> solution;
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> system_rhs;
  };



  template <int dim, int fe_degree>
  HelmholtzProblem<dim, fe_degree>::HelmholtzProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator)
    , dof_handler(triangulation)
    , fe(fe_degree)
  {}



  template <int dim, int fe_degree>
  HelmholtzProblem<dim, fe_degree>::~HelmholtzProblem()
  {
    dof_handler.clear();
  }



  template <int dim, int fe_degree>
  void HelmholtzProblem<dim, fe_degree>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);
    constraints.close();
    system_matrix.reset(new HelmholtzOperator(dof_handler, constraints));

    solution.reinit(locally_owned_dofs);
    system_rhs.reinit(locally_owned_dofs);
  }



} // namespace Step85
