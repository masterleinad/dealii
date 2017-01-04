// ---------------------------------------------------------------------
//
// Copyright (C) 2016 by the deal.II authors
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


// test ConstraintMatrix::shift for a ConstraintMatrix object
// initialized with an IndexSet object

#include "../tests.h"
#include <fstream>

#include <deal.II/base/index_set.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

void test()
{
  const int size = 20;

  dealii::IndexSet index_set(size);
  dealii::IndexSet set1(size);
  deallog << "Create index set with entries: ";
  index_set.add_index(2);
  index_set.add_index(5);
  index_set.add_index(8);
  index_set.add_index(12);
  index_set.add_index(15);
  index_set.add_index(18);
  index_set.print(deallog);

  deallog << "Create ConstraintMatrix with constraints u(2)=.5*u(5), u(5)=.7*u(8)" << std::endl;
  dealii::ConstraintMatrix constraints1(index_set);
  constraints1.add_line(2);
  constraints1.add_entry(2, 5, .5);
  constraints1.add_line(5);
  constraints1.add_entry(5, 8, .7);
  dealii::ConstraintMatrix constraints2(constraints1);
  constraints1.print(deallog.get_file_stream());
  constraints1.shift(size/2);
  constraints1.close();
  deallog << "Shifted constraints" << std::endl;
  constraints1.print(deallog.get_file_stream());
  constraints2.merge(constraints1);
  deallog << "Shifted and merged constraints" << std::endl;
  constraints2.print(deallog.get_file_stream());
  constraints2.close();
  deallog << "Close" << std::endl;
  constraints2.print(deallog.get_file_stream());

  Vector<double> vec(size);
  for (unsigned int i=0; i<size; ++i)
    vec(i) = i;
  constraints2.distribute(vec);
  vec.print(deallog.get_file_stream(), 3, true, false);
}

int main()
{
  std::ofstream logfile("output");
  deallog.attach(logfile);
  deallog.threshold_double(1.e-10);

  test();

  return 0;
}
