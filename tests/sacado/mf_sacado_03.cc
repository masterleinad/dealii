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

#include <deal.II/lac/vector_memory.templates.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/differentiation/ad.h>

#include <iostream>

#include <deal.II/matrix_free/matrix_free.templates.h>
#include <deal.II/matrix_free/mapping_info.templates.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>

using ADNumberType = Sacado::Fad::DFad<double>;

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues () : Function<dim>() {}
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
                                   const unsigned int /*component*/) const
{
  return p.square();
}

template <int dim, int fe_degree, typename Number, typename VectorType=Vector<Number>, int n_q_points_1d=fe_degree+1>
class MatrixFreeTangentOperator
{
public:
  typedef VectorizedArray<Number> vector_t;

  void reinit(const MatrixFree<dim,typename VectorType::value_type> &data_in,
              const VectorType *const solution)
  {
    data = &data_in;
    solution_ptr = solution;
  }

  const double alpha = 10.;

  const VectorType *solution_ptr;

  void
  schloegl_operator (const MatrixFree<dim,typename VectorType::value_type> &data,
                     VectorType                                            &dst,
                     const VectorType                                      &src,
                     const std::pair<unsigned int,unsigned int>            &cell_range) const
  {
    using ADNumberType = Sacado::Fad::DFad<double>;

    FEEvaluation<dim,fe_degree,n_q_points_1d,1,typename VectorType::value_type> fe_eval (data);
    FEEvaluation<dim,fe_degree,n_q_points_1d,1,typename VectorType::value_type> fe_eval_linear (data);

    const unsigned int n_q_points = fe_eval.n_q_points;

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        fe_eval.reinit (cell);
        fe_eval_linear.reinit (cell);
        fe_eval.read_dof_values (src);
        fe_eval_linear.read_dof_values (*solution_ptr);

        fe_eval.evaluate (true, true, false);
        fe_eval_linear.evaluate (true, true, false);
        for (unsigned int q=0; q<n_q_points; ++q)
          {
            VectorizedArray<typename VectorType::value_type> residual_q;
            for (unsigned int vec=0; vec<VectorizedArray<Number>::n_array_elements; ++vec)
              {
                const ADNumberType u (1, 0, fe_eval_linear.get_value(q)[vec]);
                const ADNumberType residual_q_u = 3.*u*u*u+alpha*u;
                residual_q[vec] = residual_q_u.dx(0);
              }
            fe_eval.submit_value (residual_q*fe_eval.get_value(q),q);
            fe_eval.submit_gradient (fe_eval.get_gradient(q),q);
          }
        fe_eval.integrate (true,true);

        fe_eval.distribute_local_to_global (dst);
      }
  }

  void vmult (VectorType       &dst,
              const VectorType &src) const
  {
    dst = 0;
    const std::function<void(const MatrixFree<dim,typename VectorType::value_type> &,
                             VectorType &,
                             const VectorType &,
                             const std::pair<unsigned int,unsigned int> &)>
    wrap = [&](const MatrixFree<dim,typename VectorType::value_type> &mf,
               VectorType &dst,
               const VectorType &src,
               const std::pair<unsigned int,unsigned int> &pair)
    {
      this->schloegl_operator(mf, dst, src, pair);
    };
    data->cell_loop (wrap, dst, src);
  }

private:
  const MatrixFree<dim,typename VectorType::value_type> *data;
};


template <int dim, int fe_degree, typename Number, typename VectorType=Vector<Number>, int n_q_points_1d=fe_degree+1>
class MatrixFreeResidualOperator
{
public:
  typedef VectorizedArray<Number> vector_t;

  void reinit(const MatrixFree<dim,typename VectorType::value_type> &data_in)
  {
    data = &data_in;
  }

  const double alpha = 10.;

  void
  schloegl_operator (const MatrixFree<dim,typename VectorType::value_type> &data,
                     VectorType                                            &dst,
                     const VectorType                                      &src,
                     const std::pair<unsigned int,unsigned int>            &cell_range) const
  {
    FEEvaluation<dim,fe_degree,n_q_points_1d,1,typename VectorType::value_type> fe_eval (data);

    const unsigned int n_q_points = fe_eval.n_q_points;

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        fe_eval.reinit (cell);
        fe_eval.read_dof_values_plain (src);

        fe_eval.evaluate (true, true, false);
        for (unsigned int q=0; q<n_q_points; ++q)
          {
            const VectorizedArray<Number> u = fe_eval.get_value(q);
            fe_eval.submit_value (3.*u*u*u+alpha*u,q);
            fe_eval.submit_gradient (fe_eval.get_gradient(q),q);
          }
        fe_eval.integrate (true,true);

        fe_eval.distribute_local_to_global (dst);
      }
  }

  void vmult (VectorType       &dst,
              const VectorType &src) const
  {
    dst = 0;
    const std::function<void(const MatrixFree<dim,typename VectorType::value_type> &,
                             VectorType &,
                             const VectorType &,
                             const std::pair<unsigned int,unsigned int> &)>
    wrap = [&](const MatrixFree<dim,typename VectorType::value_type> &mf,
               VectorType &dst,
               const VectorType &src,
               const std::pair<unsigned int,unsigned int> &pair)
    {
      this->schloegl_operator(mf,dst,src, pair);
    };
    data->cell_loop (wrap, dst, src);
  }

private:
  const MatrixFree<dim,typename VectorType::value_type> *data;
};


template <int dim, int fe_degree, typename number, int n_q_points_1d>
class TestSchloegl
{
public:

  TestSchloegl ()
    :fe(fe_degree)
  {
    GridGenerator::hyper_cube (tria, -1., 1.);
    tria.refine_global(4);

    dof_handler.initialize(tria, fe);

    ConstraintMatrix constraints;
    constraints.clear();
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              BoundaryValues<dim>(),
                                              constraints);
    constraints.close();

    homogeneous_constraints.clear();
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              Functions::ZeroFunction<dim>(),
                                              homogeneous_constraints);
    homogeneous_constraints.close();

    solution.reinit(dof_handler.n_dofs());
    solution_increment.reinit(dof_handler.n_dofs());
    residual_vector.reinit(dof_handler.n_dofs());

    constraints.distribute(solution);

    MappingQGeneric<dim> mapping(dof_handler.get_fe().degree);
    MatrixFree<dim,double> mf_data;
    {
      const QGauss<1> quad (n_q_points_1d);
      typename MatrixFree<dim, double>::AdditionalData data;
      data.tasks_parallel_scheme =
        MatrixFree<dim, double>::AdditionalData::partition_partition;
      data.tasks_block_size = 7;

      mf_data.reinit (mapping, dof_handler, homogeneous_constraints, quad, data);
    }

    mf_residual.reinit(mf_data);
    mf_tangent.reinit(mf_data, &solution);
    const unsigned int n_iterations =  10.;
    for (unsigned int iteration=0; iteration<n_iterations; ++iteration)
      {
        solution_increment = 0.;
        // Construct residula vector;
        mf_residual.vmult(residual_vector, solution);
        residual_vector *= -1;

        SolverControl solver_control(1000, 1.e-6*residual_vector.l2_norm());
        SolverCG<> solver_cg(solver_control);

        solver_cg.solve(mf_tangent, solution_increment, residual_vector, PreconditionIdentity());
        homogeneous_constraints.distribute(solution_increment);
        solution += solution_increment;

  /*      DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (solution, "solution");
        data_out.build_patches ();
        std::ofstream output ("solution-2d"+Utilities::int_to_string(iteration)+".vtk");
        data_out.write_vtk (output);*/

        deallog << "Solution norm: " << solution.l2_norm() << std::endl;
        deallog << "Residual norm: " << residual_vector.l2_norm() << std::endl;
        deallog << "Solution increment norm: " << solution_increment.l2_norm() << std::endl;
      }
  }

  Triangulation<dim> tria;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;
  ConstraintMatrix homogeneous_constraints;

  Vector<double> solution;
  Vector<double> solution_increment;
  Vector<double> residual_vector;

  MatrixFreeResidualOperator<dim, fe_degree, double> mf_residual;
  MatrixFreeTangentOperator<dim, fe_degree, double>  mf_tangent;
};

int main ()
{
  MultithreadInfo::set_thread_limit(1);
  initlog();

  {
    deallog.push("2d");
    TestSchloegl<2,1,double,2> test;
    deallog.pop();
  }
}
