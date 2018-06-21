// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2017 by the deal.II authors
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



// it used to happen that if we destroyed logstream (and presumably
// all objects of the same type) that whatever we had put into with
// operator<< after the last use of std::endl was lost. make sure that
// that isn't the case anymore: logstream should flush whatever it has
// left over when it is destroyed


#include <limits>

#include "../tests.h"


int
main()
{
  std::ofstream logfile("output");
  deallog.attach(logfile);

  {
    LogStream log;

    log.attach(logfile);
    log.log_thread_id(false);

    log << "This should be printed!";
  }
  deallog << "OK" << std::endl;
}
