// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2016 by the deal.II authors
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


#include "../tests.h"
#include <deal.II/base/logstream.h>
#include <deal.II/base/std_cxx11/unique_ptr.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>

#include <fstream>
#include <iomanip>
#include <iomanip>
#include <string>


template <int dim>
parallel::distributed::Triangulation<dim> *make_tria ()
{
  parallel::distributed::Triangulation<dim> *tria = new parallel::distributed::Triangulation<dim>(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(*tria, 0., 1.);
  tria->refine_global (1);
  for (int i=0; i<2; ++i)
    {
      tria->begin_active()->set_refine_flag();
      tria->execute_coarsening_and_refinement ();
    }
  return tria;
}



template <int dim>
DoFHandler<dim> *make_dof_handler (const parallel::distributed::Triangulation<dim> &tria,
                                   const FiniteElement<dim> &fe)
{
  DoFHandler<dim> *dof_handler = new DoFHandler<dim>(tria);
  dof_handler->distribute_dofs (fe);
  return dof_handler;
}



// output some indicators for a given vector
template <unsigned int dim, typename VectorType>
void
output_vector (const VectorType &v, const std::string &output_name, const DoFHandler<dim> &dof_handler)
{
  deallog << v.l1_norm() << ' ' << v.l2_norm() << ' ' << v.linfty_norm()
          << std::endl;

  v.print(deallog.get_file_stream());

  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (v, output_name, DataOut<dim>::type_dof_data);

  const unsigned int degree = dof_handler.get_fe().degree;
  data_out.build_patches (degree);

  const std::string filename = (output_name +
                                "." +
                                Utilities::int_to_string
                                (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD), 1));
  std::ofstream output ((filename + ".vtu").c_str());
  data_out.write_vtu (output);

}



template <int dim, typename VectorType>
void
check_this (const FiniteElement<dim> &fe1,
            const FiniteElement<dim> &fe2)
{
  deallog << std::setprecision (10);

  // only check if both elements have
  // support points. otherwise,
  // interpolation doesn't really
  // work
  if ((fe1.get_unit_support_points().size() == 0) ||
      (fe2.get_unit_support_points().size() == 0))
    return;
  //  likewise for non-primitive elements
  if (!fe1.is_primitive() || !fe2.is_primitive())
    return;
  // we need to have dof_constraints
  // for this test
  if (!fe2.constraints_are_implemented())
    return;
  // we need prolongation matrices in
  // fe2
  if (!fe2.isotropic_restriction_is_implemented())
    return;

  std_cxx11::unique_ptr<parallel::distributed::Triangulation<dim> > tria(make_tria<dim>());

  std_cxx11::unique_ptr<DoFHandler<dim> >    dof1(make_dof_handler (*tria, fe1));
  std_cxx11::unique_ptr<DoFHandler<dim> >    dof2(make_dof_handler (*tria, fe2));
  ConstraintMatrix cm;
  DoFTools::make_hanging_node_constraints (*dof2, cm);
  cm.close ();

  IndexSet locally_owned_dofs1 = dof1->locally_owned_dofs();
  IndexSet locally_relevant_dofs1;
  DoFTools::extract_locally_relevant_dofs (*dof1, locally_relevant_dofs1);
  IndexSet locally_owned_dofs2 = dof2->locally_owned_dofs();
  IndexSet locally_relevant_dofs2;
  DoFTools::extract_locally_relevant_dofs (*dof2, locally_relevant_dofs2);

  VectorType in_ghosted (locally_owned_dofs1, locally_relevant_dofs1, MPI_COMM_WORLD);
  VectorType in_distributed (locally_owned_dofs1, MPI_COMM_WORLD);
  VectorType out_distributed (locally_owned_dofs2, MPI_COMM_WORLD);
  VectorType out_ghosted (locally_owned_dofs2, locally_relevant_dofs2, MPI_COMM_WORLD);

  locally_owned_dofs1.print(std::cout);
  in_ghosted.locally_owned_elements().print(std::cout);
  locally_owned_dofs2.print(std::cout);
  out_ghosted.locally_owned_elements().print(std::cout);

  IndexSet::ElementIterator it = locally_owned_dofs1.begin();
  for (unsigned int i=0; it != locally_owned_dofs1.end(); ++it, ++i)
    in_distributed(*it) = i;
  in_distributed.compress(VectorOperation::insert);
  in_ghosted = in_distributed;

  output_vector<dim, VectorType> (in_ghosted, std::string("in"), *dof1);
  FETools::extrapolate_parallel (*dof1, in_ghosted, *dof2, cm, out_distributed);
  out_ghosted = out_distributed;
  output_vector<dim, VectorType> (out_ghosted, std::string("out"), *dof2);
}
