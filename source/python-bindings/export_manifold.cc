// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 by the deal.II authors
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

#include <deal.II/python-bindings/manifold_wrapper.h>

#include <boost/python.hpp>

DEAL_II_NAMESPACE_OPEN

namespace python
{
  BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(create_cylindrical_overloads,
                                         create_cylindrical,
                                         0,
                                         2)


  const char create_spherical_docstring[] =
    " Create spherical manifold with a given center point.      \n";


  const char create_polar_docstring[] =
    " Create polar manifold with a given center point.          \n";


  const char create_function_docstring[] =
    " Create manifold with the given python push forward and    \n"
    " pull back functions.                                      \n";


  const char create_function_string_docstring[] =
    " Create manifold with given string expression for the push \n"
    " forward and pull back functions.                          \n";


  const char create_cylindrical_docstring[] =
    " Create cylindrical manifold along a given axis            \n"
    " (0 - x, 1 - y, 2 - z).                                    \n";



  void
  export_manifold()
  {
    boost::python::class_<ManifoldWrapper>(
      "Manifold",
      boost::python::init<const int, const int>(
        boost::python::args("dim", "spacedim")))
      .def("create_spherical",
           &ManifoldWrapper::create_spherical,
           create_spherical_docstring,
           boost::python::args("self", "center"))
      .def("create_polar",
           &ManifoldWrapper::create_polar,
           create_polar_docstring,
           boost::python::args("self", "center"))
      .def("create_cylindrical",
           &ManifoldWrapper::create_cylindrical,
           create_cylindrical_overloads(
             boost::python::args("self", "axis", "tolerance"),
             create_cylindrical_docstring))
      .def("create_function",
           &ManifoldWrapper::create_function,
           create_function_docstring,
           boost::python::args("self", "push_forward", "pull_back"))
      .def("create_function_string",
           &ManifoldWrapper::create_function_string,
           create_function_string_docstring,
           boost::python::args("self", "push_forward", "pull_back"));
  }
} // namespace python

DEAL_II_NAMESPACE_CLOSE
