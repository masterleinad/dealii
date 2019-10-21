// ---------------------------------------------------------------------
//
// Copyright (C) 2013 - 2018 by the deal.II authors
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



// this function tests the correctness of the implementation of matrix free
// matrix-vector products by comparing with the result of deal.II sparse
// matrix for MG DoFHandler on a hyperball mesh with no hanging nodes but
// homogeneous Dirichlet conditions

#include <deal.II/base/function.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/la_vector.h>

#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"
#include "matrix_vector_mf.h"



template <int dim, int fe_degree>
void
test()
{
  const SphericalManifold<dim> manifold;
  Triangulation<dim>           tria(
    Triangulation<dim>::limit_level_difference_at_vertices);
  GridGenerator::hyper_ball(tria);
  for (const auto &cell : tria.active_cell_iterators())
    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      if (cell->at_boundary(f))
        cell->face(f)->set_all_manifold_ids(0);
  tria.set_manifold(0, manifold);
  tria.refine_global(5 - dim);

  FE_Q<dim> fe(fe_degree);

  // setup DoFs
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  dof.distribute_mg_dofs();
  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(dof,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  // std::cout << "Number of cells: " <<
  // dof.get_triangulation().n_active_cells() << std::endl; std::cout << "Number
  // of degrees of freedom: " << dof.n_dofs() << std::endl;

  // set up MatrixFree
  MappingQGeneric<dim> mapping(fe_degree);
  QGauss<1>            quad(fe_degree + 1);
  typename CUDAWrappers::MatrixFree<dim, double>::AdditionalData
    additional_data;
  additional_data.mapping_update_flags = update_values | update_gradients |
                                         update_JxW_values |
                                         update_quadrature_points;
  CUDAWrappers::MatrixFree<dim> mf_data;
  mf_data.reinit(mapping, dof, constraints, quad, additional_data);
  SparsityPattern      sparsity;
  SparseMatrix<double> system_matrix;
  {
    DynamicSparsityPattern csp(dof.n_dofs(), dof.n_dofs());
    DoFTools::make_sparsity_pattern(static_cast<const DoFHandler<dim> &>(dof),
                                    csp,
                                    constraints,
                                    false);
    sparsity.copy_from(csp);
  }
  system_matrix.reinit(sparsity);

  // setup MG levels
  const unsigned int                       nlevels = tria.n_levels();
  typedef CUDAWrappers::MatrixFree<dim>    MatrixFreeTestType;
  MGLevelObject<MatrixFreeTestType>        mg_matrices;
  MGLevelObject<AffineConstraints<double>> mg_constraints;
  MGLevelObject<SparsityPattern>           mg_sparsities;
  MGLevelObject<SparseMatrix<double>>      mg_ref_matrices;
  mg_matrices.resize(0, nlevels - 1);
  mg_constraints.resize(0, nlevels - 1);
  mg_sparsities.resize(0, nlevels - 1);
  mg_ref_matrices.resize(0, nlevels - 1);

  std::map<types::boundary_id, const Function<dim> *> dirichlet_boundary;
  Functions::ZeroFunction<dim> homogeneous_dirichlet_bc(1);
  dirichlet_boundary[0] = &homogeneous_dirichlet_bc;
  std::vector<std::set<types::global_dof_index>> boundary_indices(nlevels);
  MGTools::make_boundary_list(dof, dirichlet_boundary, boundary_indices);
  for (unsigned int level = 0; level < nlevels; ++level)
    {
      std::set<types::global_dof_index>::iterator bc_it =
        boundary_indices[level].begin();
      for (; bc_it != boundary_indices[level].end(); ++bc_it)
        mg_constraints[level].add_line(*bc_it);
      mg_constraints[level].close();
      typename CUDAWrappers::MatrixFree<dim>::AdditionalData data;
      data.mapping_update_flags = update_values | update_gradients |
                                         update_JxW_values |
                                         update_quadrature_points;
      data.level_mg_handler = level;
      mg_matrices[level].reinit(
        mapping, dof, mg_constraints[level], quad, data);

      DynamicSparsityPattern csp;
      csp.reinit(dof.n_dofs(level), dof.n_dofs(level));
      MGTools::make_sparsity_pattern(dof, csp, level);
      mg_sparsities[level].copy_from(csp);
      mg_ref_matrices[level].reinit(mg_sparsities[level]);
    }

  // assemble sparse matrix with (\nabla v,
  // \nabla u) + (v, 10 * u) on the actual
  // discretization and on all levels
  {
    QGauss<dim>   quad(fe_degree + 1);
    FEValues<dim> fe_values(
      mapping, fe, quad, update_values | update_gradients | update_JxW_values);
    const unsigned int n_quadrature_points = quad.size();
    const unsigned int dofs_per_cell       = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof.active_cell_iterators())
      {
        cell_matrix = 0;
        fe_values.reinit(cell);

        for (unsigned int q_point = 0; q_point < n_quadrature_points; ++q_point)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) += ((fe_values.shape_grad(i, q_point) *
                                         fe_values.shape_grad(j, q_point) +
                                       10. * fe_values.shape_value(i, q_point) *
                                         fe_values.shape_value(j, q_point)) *
                                      fe_values.JxW(q_point));
            }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               system_matrix);
      }

    // now to the MG assembly
    typename DoFHandler<dim>::cell_iterator cellm = dof.begin(),
                                            endcm = dof.end();
    for (; cellm != endcm; ++cellm)
      {
        cell_matrix = 0;
        fe_values.reinit(cellm);

        for (unsigned int q_point = 0; q_point < n_quadrature_points; ++q_point)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                cell_matrix(i, j) += ((fe_values.shape_grad(i, q_point) *
                                         fe_values.shape_grad(j, q_point) +
                                       10. * fe_values.shape_value(i, q_point) *
                                         fe_values.shape_value(j, q_point)) *
                                      fe_values.JxW(q_point));
            }
        cellm->get_mg_dof_indices(local_dof_indices);
        mg_constraints[cellm->level()].distribute_local_to_global(
          cell_matrix, local_dof_indices, mg_ref_matrices[cellm->level()]);
      }
  }

  // fill a right hand side vector with random
  // numbers in unconstrained degrees of freedom
  LinearAlgebra::CUDAWrappers::Vector<double> src(dof.n_dofs());
  Vector<double>                              in_host(dof.n_dofs());
  LinearAlgebra::CUDAWrappers::Vector<double> result_mf(src);
  Vector<double>                              result_spmv(in_host);

  LinearAlgebra::ReadWriteVector<double> rw_in(dof.n_dofs());
  for (unsigned int i = 0; i < dof.n_dofs(); ++i)
    {
      if (constraints.is_constrained(i) == false)
        {
          rw_in(i)   = random_value<double>();
          in_host(i) = rw_in(i);
        }
    }
  src.import(rw_in, VectorOperation::insert);

  // now perform matrix-vector product and check
  // its correctness
  system_matrix.vmult(result_spmv, in_host);

  const unsigned int coef_size =
    tria.n_active_cells() * std::pow(fe_degree + 1, dim);
  MatrixFreeTest<dim,
                 fe_degree,
                 double,
                 LinearAlgebra::CUDAWrappers::Vector<double>>
    mf(mf_data, coef_size);

  mf.vmult(result_mf, src);

  LinearAlgebra::ReadWriteVector<double> rw_out(dof.n_dofs());
  rw_out.import(result_mf, VectorOperation::insert);
  for (unsigned int i = 0; i < result_spmv.size(); ++i)
    result_spmv(i) -= rw_out(i);
  const double diff_norm = result_spmv.linfty_norm();
  deallog << "Norm of difference active: " << diff_norm << std::endl;

  for (unsigned int level = 0; level < nlevels; ++level)
    {
      LinearAlgebra::CUDAWrappers::Vector<double> level_src(dof.n_dofs(level));
      Vector<double> level_in_host(dof.n_dofs(level));
      LinearAlgebra::CUDAWrappers::Vector<double> level_result_mf(level_src);
      Vector<double> level_result_spmv(level_in_host);

      LinearAlgebra::ReadWriteVector<double> level_rw_in(dof.n_dofs());
      for (unsigned int i = 0; i < dof.n_dofs(level); ++i)
        {
          if (mg_constraints[level].is_constrained(i) == false)
            {
              level_rw_in(i)   = random_value<double>();
              level_in_host(i) = level_rw_in(i);
            }
        }
      level_src.import(level_rw_in, VectorOperation::insert);

      // now perform matrix-vector product and check
      // its correctness
      mg_ref_matrices[level].vmult(level_result_spmv, level_in_host);
      const unsigned int level_coef_size =
        tria.n_active_cells(level) * std::pow(fe_degree + 1, dim);
      MatrixFreeTest<dim,
                     fe_degree,
                     double,
                     LinearAlgebra::CUDAWrappers::Vector<double>>
        mf_lev(mg_matrices[level], level_coef_size);
      mf_lev.vmult(level_result_mf, level_src);

      LinearAlgebra::ReadWriteVector<double> level_rw_out(dof.n_dofs(level));
      for (unsigned int i = 0; i < level_result_spmv.size(); ++i)
        level_result_spmv(i) -= level_rw_out(i);
      const double diff_norm = level_result_spmv.linfty_norm();
      deallog << "Norm of difference MG level " << level << ": " << diff_norm
              << std::endl;
    }
  deallog << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  unsigned int myid = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  deallog.push(Utilities::int_to_string(myid));

  init_cuda(true);

  if (myid == 0)
    {
      initlog();
      deallog << std::setprecision(4);

      deallog.push("2d");
      test<2, 1>();
      deallog.pop();

      deallog.push("3d");
      test<3, 1>();
      test<3, 2>();
      deallog.pop();
    }
  else
    {
      test<2, 1>();
      test<3, 1>();
      test<3, 2>();
    }
}
