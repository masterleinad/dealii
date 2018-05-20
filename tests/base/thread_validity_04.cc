// ---------------------------------------------------------------------
//
// Copyright (C) 2008 - 2017 by the deal.II authors
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

// test that objects that can't be copied aren't copied when passed to a new
// thread by reference
//
// this is a variant that makes sure that member functions of objects that
// can't be copied aren't called on copies. this test is for const member
// functions

#include "../tests.h"

#include <deal.II/base/thread_management.h>

struct X
{
  X(int i) : i(i)
  {}
  int i;

  void
  execute() const
  {
    Assert(i == 42, ExcInternalError());
    deallog << "OK" << std::endl;
  }

private:
  X(const X&);
  X&
  operator=(const X&);
};

void
test()
{
  const X x(42);
  Threads::Thread<void> t = Threads::new_thread(&X::execute, x);
  t.join();
}

int
main()
{
  initlog();

  test();
}
