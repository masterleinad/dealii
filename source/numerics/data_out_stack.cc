// ---------------------------------------------------------------------
//
// Copyright (C) 1999 - 2018 by the deal.II authors
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

#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/tria_iterator.h>

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out_stack.h>

#include <sstream>

DEAL_II_NAMESPACE_OPEN


template <int dim, int spacedim, typename DoFHandlerType>
std::size_t
DataOutStack<dim, spacedim, DoFHandlerType>::DataVector::memory_consumption()
  const
{
  return (MemoryConsumption::memory_consumption(data) +
          MemoryConsumption::memory_consumption(names));
}



template <int dim, int spacedim, typename DoFHandlerType>
void
DataOutStack<dim, spacedim, DoFHandlerType>::new_parameter_value(
  const double p,
  const double dp)
{
  parameter      = p;
  parameter_step = dp;

  // check whether the user called finish_parameter_value() at the end of the
  // previous parameter step
  //
  // this is to prevent serious waste of memory
  for (typename std::vector<DataVector>::const_iterator i = dof_data.begin();
       i != dof_data.end();
       ++i)
    Assert(i->data.size() == 0, ExcDataNotCleared());
  for (typename std::vector<DataVector>::const_iterator i = cell_data.begin();
       i != cell_data.end();
       ++i)
    Assert(i->data.size() == 0, ExcDataNotCleared());
}


template <int dim, int spacedim, typename DoFHandlerType>
void
DataOutStack<dim, spacedim, DoFHandlerType>::attach_dof_handler(
  const DoFHandlerType &dof)
{
  // Check consistency of redundant
  // template parameter
  Assert(dim == DoFHandlerType::dimension,
         ExcDimensionMismatch(dim, DoFHandlerType::dimension));

  dof_handler = &dof;
}


template <int dim, int spacedim, typename DoFHandlerType>
void
DataOutStack<dim, spacedim, DoFHandlerType>::declare_data_vector(
  const std::string &name,
  const VectorType   vector_type)
{
  std::vector<std::string> names;
  names.push_back(name);
  declare_data_vector(names, vector_type);
}


template <int dim, int spacedim, typename DoFHandlerType>
void
DataOutStack<dim, spacedim, DoFHandlerType>::declare_data_vector(
  const std::vector<std::string> &names,
  const VectorType                vector_type)
{
  // make sure this function is
  // not called after some parameter
  // values have already been
  // processed
  Assert(patches.size() == 0, ExcDataAlreadyAdded());

#ifdef DEBUG
  // also make sure that no name is
  // used twice
  for (const auto &name : names)
    {
      for (const auto &data_set : dof_data)
        for (const auto &data_set_name : data_set.names)
          Assert(name != data_set_name, ExcNameAlreadyUsed(name));

      for (const auto &data_set : cell_data)
        for (const auto &data_set_name : data_set.names)
          Assert(name != data_set_name, ExcNameAlreadyUsed(name));
    };
#endif

  switch (vector_type)
    {
      case dof_vector:
        dof_data.emplace_back();
        dof_data.back().names = names;
        break;

      case cell_vector:
        cell_data.emplace_back();
        cell_data.back().names = names;
        break;
    };
}


template <int dim, int spacedim, typename DoFHandlerType>
template <typename number>
void
DataOutStack<dim, spacedim, DoFHandlerType>::add_data_vector(
  const Vector<number> &vec,
  const std::string &   name)
{
  const unsigned int n_components = dof_handler->get_fe(0).n_components();

  std::vector<std::string> names;
  // if only one component or vector
  // is cell vector: we only need one
  // name
  if ((n_components == 1) ||
      (vec.size() == dof_handler->get_triangulation().n_active_cells()))
    {
      names.resize(1, name);
    }
  else
    // otherwise append _i to the
    // given name
    {
      names.resize(n_components);
      for (unsigned int i = 0; i < n_components; ++i)
        {
          std::ostringstream namebuf;
          namebuf << '_' << i;
          names[i] = name + namebuf.str();
        }
    }

  add_data_vector(vec, names);
}


template <int dim, int spacedim, typename DoFHandlerType>
template <typename number>
void
DataOutStack<dim, spacedim, DoFHandlerType>::add_data_vector(
  const Vector<number> &          vec,
  const std::vector<std::string> &names)
{
  Assert(dof_handler != nullptr,
         Exceptions::DataOutImplementation::ExcNoDoFHandlerSelected());
  // either cell data and one name,
  // or dof data and n_components names
  Assert(((vec.size() == dof_handler->get_triangulation().n_active_cells()) &&
          (names.size() == 1)) ||
           ((vec.size() == dof_handler->n_dofs()) &&
            (names.size() == dof_handler->get_fe(0).n_components())),
         Exceptions::DataOutImplementation::ExcInvalidNumberOfNames(
           names.size(), dof_handler->get_fe(0).n_components()));
  for (const auto &name : names)
    {
      (void)name;
      Assert(name.find_first_not_of("abcdefghijklmnopqrstuvwxyz"
                                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                    "0123456789_<>()") == std::string::npos,
             Exceptions::DataOutImplementation::ExcInvalidCharacter(
               name,
               name.find_first_not_of("abcdefghijklmnopqrstuvwxyz"
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                      "0123456789_<>()")));
    }

  if (vec.size() == dof_handler->n_dofs())
    {
      auto data_vector = dof_data.begin();
      for (; data_vector != dof_data.end(); ++data_vector)
        if (data_vector->names == names)
          {
            data_vector->data.reinit(vec.size());
            std::copy(vec.begin(), vec.end(), data_vector->data.begin());
            return;
          };

      // ok. not found. there is a
      // slight chance that
      // n_dofs==n_cells, so only
      // bomb out if the next if
      // statement will not be run
      if (dof_handler->n_dofs() !=
          dof_handler->get_triangulation().n_active_cells())
        Assert(false, ExcVectorNotDeclared(names[0]));
    }

  // search cell data
  if ((vec.size() != dof_handler->n_dofs()) ||
      (dof_handler->n_dofs() ==
       dof_handler->get_triangulation().n_active_cells()))
    {
      auto data_vector = cell_data.begin();
      for (; data_vector != cell_data.end(); ++data_vector)
        if (data_vector->names == names)
          {
            data_vector->data.reinit(vec.size());
            std::copy(vec.begin(), vec.end(), data_vector->data.begin());
            return;
          };
      Assert(false, ExcVectorNotDeclared(names[0]));
    };

  // we have either return or Assert
  // statements above, so shouldn't
  // get here!
  Assert(false, ExcInternalError());
}


template <int dim, int spacedim, typename DoFHandlerType>
void
DataOutStack<dim, spacedim, DoFHandlerType>::build_patches(
  const unsigned int nnnn_subdivisions)
{
  // this is mostly copied from the
  // DataOut class
  unsigned int n_subdivisions =
    (nnnn_subdivisions != 0) ? nnnn_subdivisions : this->default_subdivisions;

  Assert(n_subdivisions >= 1,
         Exceptions::DataOutImplementation::ExcInvalidNumberOfSubdivisions(
           n_subdivisions));
  Assert(dof_handler != nullptr,
         Exceptions::DataOutImplementation::ExcNoDoFHandlerSelected());

  this->validate_dataset_names();

  const unsigned int n_components = dof_handler->get_fe(0).n_components();
  const unsigned int n_datasets =
    dof_data.size() * n_components + cell_data.size();

  // first count the cells we want to
  // create patches of and make sure
  // there is enough memory for that
  unsigned int n_patches = 0;
  for (typename DoFHandlerType::active_cell_iterator cell =
         dof_handler->begin_active();
       cell != dof_handler->end();
       ++cell)
    ++n_patches;


  // before we start the loop:
  // create a quadrature rule that
  // actually has the points on this
  // patch, and an object that
  // extracts the data on each
  // cell to these points
  QTrapez<1>     q_trapez;
  QIterated<dim> patch_points(q_trapez, n_subdivisions);

  // create collection objects from
  // single quadratures,
  // and finite elements. if we have
  // an hp DoFHandler,
  // dof_handler.get_fe() returns a
  // collection of which we do a
  // shallow copy instead
  const hp::QCollection<dim>   q_collection(patch_points);
  const hp::FECollection<dim> &fe_collection = dof_handler->get_fe_collection();

  hp::FEValues<dim> x_fe_patch_values(fe_collection,
                                      q_collection,
                                      update_values);

  const unsigned int          n_q_points = patch_points.size();
  std::vector<double>         patch_values(n_q_points);
  std::vector<Vector<double>> patch_values_system(n_q_points,
                                                  Vector<double>(n_components));

  // add the required number of
  // patches. first initialize a template
  // patch with n_q_points (in the plane
  // of the cells) times n_subdivisions+1 (in
  // the time direction) points
  dealii::DataOutBase::Patch<dim + 1, dim + 1> default_patch;
  default_patch.n_subdivisions = n_subdivisions;
  default_patch.data.reinit(n_datasets, n_q_points * (n_subdivisions + 1));
  patches.insert(patches.end(), n_patches, default_patch);

  // now loop over all cells and
  // actually create the patches
  auto         patch       = patches.begin() + (patches.size() - n_patches);
  unsigned int cell_number = 0;
  for (typename DoFHandlerType::active_cell_iterator cell =
         dof_handler->begin_active();
       cell != dof_handler->end();
       ++cell, ++patch, ++cell_number)
    {
      Assert(cell->is_locally_owned(), ExcNotImplemented());

      Assert(patch != patches.end(), ExcInternalError());

      // first fill in the vertices of the patch

      // Patches are organized such
      // that the parameter direction
      // is the last
      // coordinate. Thus, vertices
      // are two copies of the space
      // patch, one at parameter-step
      // and one at parameter.
      switch (dim)
        {
          case 1:
            patch->vertices[0] =
              Point<dim + 1>(cell->vertex(0)(0), parameter - parameter_step);
            patch->vertices[1] =
              Point<dim + 1>(cell->vertex(1)(0), parameter - parameter_step);
            patch->vertices[2] = Point<dim + 1>(cell->vertex(0)(0), parameter);
            patch->vertices[3] = Point<dim + 1>(cell->vertex(1)(0), parameter);
            break;

          case 2:
            patch->vertices[0] = Point<dim + 1>(cell->vertex(0)(0),
                                                cell->vertex(0)(1),
                                                parameter - parameter_step);
            patch->vertices[1] = Point<dim + 1>(cell->vertex(1)(0),
                                                cell->vertex(1)(1),
                                                parameter - parameter_step);
            patch->vertices[2] = Point<dim + 1>(cell->vertex(2)(0),
                                                cell->vertex(2)(1),
                                                parameter - parameter_step);
            patch->vertices[3] = Point<dim + 1>(cell->vertex(3)(0),
                                                cell->vertex(3)(1),
                                                parameter - parameter_step);
            patch->vertices[4] =
              Point<dim + 1>(cell->vertex(0)(0), cell->vertex(0)(1), parameter);
            patch->vertices[5] =
              Point<dim + 1>(cell->vertex(1)(0), cell->vertex(1)(1), parameter);
            patch->vertices[6] =
              Point<dim + 1>(cell->vertex(2)(0), cell->vertex(2)(1), parameter);
            patch->vertices[7] =
              Point<dim + 1>(cell->vertex(3)(0), cell->vertex(3)(1), parameter);
            break;

          default:
            Assert(false, ExcNotImplemented());
        };


      // now fill in the data values.
      // note that the required order is
      // with highest coordinate running
      // fastest, we need to enter each
      // value (n_subdivisions+1) times
      // in succession
      if (n_datasets > 0)
        {
          x_fe_patch_values.reinit(cell);
          const FEValues<dim> &fe_patch_values =
            x_fe_patch_values.get_present_fe_values();

          // first fill dof_data
          for (unsigned int dataset = 0; dataset < dof_data.size(); ++dataset)
            {
              if (n_components == 1)
                {
                  fe_patch_values.get_function_values(dof_data[dataset].data,
                                                      patch_values);
                  for (unsigned int i = 0; i < n_subdivisions + 1; ++i)
                    for (unsigned int q = 0; q < n_q_points; ++q)
                      patch->data(dataset, q + n_q_points * i) =
                        patch_values[q];
                }
              else
                // system of components
                {
                  fe_patch_values.get_function_values(dof_data[dataset].data,
                                                      patch_values_system);
                  for (unsigned int component = 0; component < n_components;
                       ++component)
                    for (unsigned int i = 0; i < n_subdivisions + 1; ++i)
                      for (unsigned int q = 0; q < n_q_points; ++q)
                        patch->data(dataset * n_components + component,
                                    q + n_q_points * i) =
                          patch_values_system[q](component);
                }
            }

          // then do the cell data
          for (unsigned int dataset = 0; dataset < cell_data.size(); ++dataset)
            {
              const double value = cell_data[dataset].data(cell_number);
              for (unsigned int q = 0; q < n_q_points; ++q)
                for (unsigned int i = 0; i < n_subdivisions + 1; ++i)
                  patch->data(dataset + dof_data.size() * n_components,
                              q * (n_subdivisions + 1) + i) = value;
            }
        }
    }
}


template <int dim, int spacedim, typename DoFHandlerType>
void
DataOutStack<dim, spacedim, DoFHandlerType>::finish_parameter_value()
{
  // release lock on dof handler
  dof_handler = nullptr;
  for (auto i = dof_data.begin(); i != dof_data.end(); ++i)
    i->data.reinit(0);

  for (auto i = cell_data.begin(); i != cell_data.end(); ++i)
    i->data.reinit(0);
}



template <int dim, int spacedim, typename DoFHandlerType>
std::size_t
DataOutStack<dim, spacedim, DoFHandlerType>::memory_consumption() const
{
  return (DataOutInterface<dim + 1>::memory_consumption() +
          MemoryConsumption::memory_consumption(parameter) +
          MemoryConsumption::memory_consumption(parameter_step) +
          MemoryConsumption::memory_consumption(dof_handler) +
          MemoryConsumption::memory_consumption(patches) +
          MemoryConsumption::memory_consumption(dof_data) +
          MemoryConsumption::memory_consumption(cell_data));
}



template <int dim, int spacedim, typename DoFHandlerType>
const std::vector<dealii::DataOutBase::Patch<dim + 1, dim + 1>> &
DataOutStack<dim, spacedim, DoFHandlerType>::get_patches() const
{
  return patches;
}



template <int dim, int spacedim, typename DoFHandlerType>
std::vector<std::string>
DataOutStack<dim, spacedim, DoFHandlerType>::get_dataset_names() const
{
  std::vector<std::string> names;
  for (auto dataset = dof_data.begin(); dataset != dof_data.end(); ++dataset)
    names.insert(names.end(), dataset->names.begin(), dataset->names.end());
  for (auto dataset = cell_data.begin(); dataset != cell_data.end(); ++dataset)
    names.insert(names.end(), dataset->names.begin(), dataset->names.end());

  return names;
}



// explicit instantiations
#include "data_out_stack.inst"


DEAL_II_NAMESPACE_CLOSE
