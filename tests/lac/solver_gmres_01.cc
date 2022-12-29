// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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

// Check the path of SolverGMRES with distributed vectors and
// OrthogonalizationStrategy::classical_gram_schmidt.


#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include "../tests.h"


struct MyDiagonalMatrix
{
  MyDiagonalMatrix(const LinearAlgebra::distributed::Vector<double> &diagonal)
    : diagonal(diagonal)
  {}

  void
  vmult(LinearAlgebra::distributed::Vector<double> &      dst,
        const LinearAlgebra::distributed::Vector<double> &src) const
  {
    dst = src;
    dst.scale(diagonal);
  }

  const LinearAlgebra::distributed::Vector<double> &diagonal;
};



SolverControl::State
monitor_norm(const unsigned int iteration,
             const double       check_value,
             const LinearAlgebra::distributed::Vector<double> &)
{
  deallog << "   estimated residual at iteration " << iteration << ": "
          << check_value << std::endl;
  return SolverControl::success;
}


int
main()
{
  initlog();

  // Create diagonal matrix with entries between 1 and 30
  DiagonalMatrix<LinearAlgebra::distributed::Vector<double>> unit_matrix;
  unit_matrix.get_vector().reinit(30);
  unit_matrix.get_vector() = 1.0;

  LinearAlgebra::distributed::Vector<double> matrix_entries(unit_matrix.m());
  for (unsigned int i = 0; i < unit_matrix.m(); ++i)
    matrix_entries(i) = i + 1;
  MyDiagonalMatrix matrix(matrix_entries);

  LinearAlgebra::distributed::Vector<double> rhs(unit_matrix.m()),
    sol(unit_matrix.m());
  rhs = 1.;

  deallog << "Solve with PreconditionIdentity: " << std::endl;
  SolverControl control(40, 1e-4);
  SolverGMRES<LinearAlgebra::distributed::Vector<double>>::AdditionalData data3(
    8);
  data3.orthogonalization_strategy =
    SolverGMRES<LinearAlgebra::distributed::Vector<double>>::AdditionalData::
      OrthogonalizationStrategy::classical_gram_schmidt;
  SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver(control,
                                                                 data3);
  solver.connect(&monitor_norm);
  solver.solve(matrix, sol, rhs, PreconditionIdentity());

  deallog << "Solve with diagonal preconditioner: " << std::endl;
  sol = 0;
  solver.solve(matrix, sol, rhs, unit_matrix);
}
