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

#include <deal.II/differentiation/ad.h>

#include <iostream>

#include <deal.II/matrix_free/matrix_free.templates.h>
#include <deal.II/matrix_free/mapping_info.templates.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>

using ADNumberType = Sacado::Fad::DFad<double>;

template <int dim, int fe_degree, typename VectorType, int n_q_points_1d, typename ADType>
void
helmholtz_operator (const MatrixFree<dim,ADType> &data,
                    VectorType                                            &dst,
                    const VectorType                                      &src,
                    const std::pair<unsigned int,unsigned int>            &cell_range)
{
  typedef typename VectorType::value_type Number;
  FEEvaluation<dim,fe_degree,n_q_points_1d,1,ADType> fe_eval (data);
  const unsigned int n_q_points = fe_eval.n_q_points;

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit (cell);
      fe_eval.read_dof_values (src);

      const unsigned int n_independent_variables = fe_eval.dofs_per_cell;
      Vector<double> vec_copy (n_independent_variables);

      for (unsigned int i=0; i<n_independent_variables; ++i)
        {
          const ADType ad (n_independent_variables, i, fe_eval.get_dof_value(i)[0].val());
          VectorizedArray<ADType> ad_vec;
          ad_vec = ad;
          fe_eval.submit_dof_value(ad_vec, i);
          vec_copy(i) = ad.val();
        }

      fe_eval.evaluate (true, true, false);
      for (unsigned int q=0; q<n_q_points; ++q)
        {
          fe_eval.submit_value (internal::NumberType<VectorizedArray<ADType>>::value(10)*fe_eval.get_value(q),q);
          fe_eval.submit_gradient (fe_eval.get_gradient(q),q);
        }
      fe_eval.integrate (true,true);

      Vector<double> result (n_independent_variables);

      for (unsigned int I=0; I<n_independent_variables; ++I)
        {
          const ADNumberType residual_I = fe_eval.get_dof_value(I)[0];
          for (unsigned int J=0; J<n_independent_variables; ++J)
              result(I)+=residual_I.dx(J)*vec_copy(J);
        }

      for (unsigned int I=0; I<n_independent_variables; ++I)
      {
          AssertThrow(std::abs(fe_eval.get_dof_value(I)[0].val() - result(I)) < 1.e-10, ExcInternalError());
          deallog <<  result(I) << std::endl;
      }

      fe_eval.distribute_local_to_global (dst);
    }
}


template <int dim, int fe_degree, typename Number, typename VectorType=Vector<Number>, int n_q_points_1d=fe_degree+1>
class MatrixFreeTest
{
public:
  typedef VectorizedArray<Number> vector_t;

  MatrixFreeTest(const MatrixFree<dim,ADNumberType> &data_in):
    data (data_in)
  {};

  void vmult (VectorType       &dst,
              const VectorType &src) const
  {
    dst = 0;
    const std::function<void(const MatrixFree<dim,ADNumberType> &,
                             VectorType &,
                             const VectorType &,
                             const std::pair<unsigned int,unsigned int> &)>
    wrap = helmholtz_operator<dim,fe_degree,VectorType,n_q_points_1d,ADNumberType>;
    data.cell_loop (wrap, dst, src);
  };

private:
  const MatrixFree<dim,ADNumberType> &data;
};




// forward declare this function. will be implemented in .cc files
template <int dim, int fe_degree>
void test ();




template <int dim, int fe_degree, typename number, int n_q_points_1d>
void do_test (const DoFHandler<dim> &dof,
              const ConstraintMatrix &constraints,
              const unsigned int     parallel_option = 0)
{

  deallog << "Testing " << dof.get_fe().get_name() << std::endl;
  if (parallel_option > 0)
    deallog << "Parallel option: " << parallel_option << std::endl;
  //std::cout << "Number of cells: " << dof.get_triangulation().n_active_cells() << std::endl;
  //std::cout << "Number of degrees of freedom: " << dof.n_dofs() << std::endl;
  //std::cout << "Number of constraints: " << constraints.n_constraints() << std::endl;

  MappingQGeneric<dim> mapping(dof.get_fe().degree);
  MatrixFree<dim,ADNumberType> mf_data;
  {
    const QGauss<1> quad (n_q_points_1d);
    typename MatrixFree<dim,ADNumberType>::AdditionalData data;
    if (parallel_option == 1)
      data.tasks_parallel_scheme =
        MatrixFree<dim,ADNumberType>::AdditionalData::partition_color;
    else if (parallel_option == 2)
      data.tasks_parallel_scheme =
        MatrixFree<dim,ADNumberType>::AdditionalData::color;
    else
      {
        Assert (parallel_option == 0, ExcInternalError());
        data.tasks_parallel_scheme =
          MatrixFree<dim,ADNumberType>::AdditionalData::partition_partition;
      }
    data.tasks_block_size = 7;

    mf_data.reinit (mapping, dof, constraints, quad, data);
  }

  MatrixFreeTest<dim,fe_degree,number,Vector<number>,n_q_points_1d> mf (mf_data);
  Vector<number> in (dof.n_dofs()), out (dof.n_dofs());
  Vector<number> in_dist (dof.n_dofs());
  Vector<number> out_dist (in_dist);

  for (unsigned int i=0; i<dof.n_dofs(); ++i)
    {
      if (constraints.is_constrained(i))
        continue;
      const double entry = i;//random_value<double>();
      in(i) = entry;
      in_dist(i) = entry;
    }

  mf.vmult (out_dist, in_dist);


  // assemble sparse matrix with (\nabla v, \nabla u) + (v, 10 * u)
  SparsityPattern sparsity;
  {
    DynamicSparsityPattern csp(dof.n_dofs(), dof.n_dofs());
    DoFTools::make_sparsity_pattern (dof, csp, constraints, true);
    sparsity.copy_from(csp);
  }
  SparseMatrix<double> sparse_matrix (sparsity);
  {
    QGauss<dim>  quadrature_formula(n_q_points_1d);

    FEValues<dim> fe_values (mapping, dof.get_fe(), quadrature_formula,
                             update_values    |  update_gradients |
                             update_JxW_values);

    const unsigned int   dofs_per_cell = dof.get_fe().dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof.begin_active(),
    endc = dof.end();
    for (; cell!=endc; ++cell)
      {
        cell_matrix = 0;
        fe_values.reinit (cell);

        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                cell_matrix(i,j) += ((fe_values.shape_grad(i,q_point) *
                                      fe_values.shape_grad(j,q_point)
                                      +
                                      10. *
                                      fe_values.shape_value(i,q_point) *
                                      fe_values.shape_value(j,q_point)) *
                                     fe_values.JxW(q_point));
            }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global (cell_matrix,
                                                local_dof_indices,
                                                sparse_matrix);
      }
  }

  sparse_matrix.vmult (out, in);
  out -= out_dist;
  const double diff_norm = out.linfty_norm() / out_dist.linfty_norm();

  deallog << "Norm of difference: " << diff_norm << std::endl << std::endl;
}


template <int dim, int fe_degree>
void test ()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube (tria);
  tria.refine_global(5-dim);

  FE_Q<dim> fe (fe_degree);
  DoFHandler<dim> dof (tria);
  dof.distribute_dofs(fe);
  ConstraintMatrix constraints;
  constraints.close();

  do_test<dim, fe_degree, double, fe_degree+1> (dof, constraints);
}

int main ()
{
  MultithreadInfo::set_thread_limit(1);
  initlog();

  {
    deallog.push("2d");
    test<2,1>();
    //test<2,2>();
    deallog.pop();
/*    deallog.push("3d");
    test<3,1>();
    test<3,2>();
    deallog.pop();*/
  }
}
