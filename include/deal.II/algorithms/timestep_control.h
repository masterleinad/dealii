// ---------------------------------------------------------------------
//
// Copyright (C) 2010 - 2017 by the deal.II authors
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

#ifndef dealii_time_step_control_h
#define dealii_time_step_control_h

#include <cstdio>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/vector_memory.h>

DEAL_II_NAMESPACE_OPEN

class ParameterHandler;

namespace Algorithms
{
  /**
   * Control class for timestepping schemes. Its main task is determining the
   * size of the next time step and the according point in the time interval.
   * Additionally, it controls writing the solution to a file.
   *
   * The size of the next time step is determined as follows:
   * <ol>
   * <li> According to the strategy, the step size is tentatively added to the
   * current time.
   * <li> If the resulting time exceeds the final time of the interval, the
   * step size is reduced in order to meet this time.
   * <li> If the resulting time is below the final time by just a fraction of
   * the step size, the step size is increased in order to meet this time.
   * <li> The resulting step size is used from the current time.
   * </ol>
   *
   * The variable @p print_step can be used to control the amount of output
   * generated by the timestepping scheme.
   */
  class TimestepControl : public Subscriptor
  {
  public:
    /**
     * The time stepping strategies. These are controlled by the value of
     * tolerance() and start_step().
     */
    enum Strategy
    {
      /**
       * Choose a uniform time step size. The step size is determined by
       * start_step(), tolerance() is ignored.
       */
      uniform,
      /**
       * Start with the time step size given by start_step() and double it in
       * every step. tolerance() is ignored.
       *
       * This strategy is intended for pseudo-timestepping schemes computing a
       * stationary limit.
       */
      doubling
    };

    /**
     * Constructor setting default values
     */
    TimestepControl(double start      = 0.,
                    double final      = 1.,
                    double tolerance  = 1.e-2,
                    double start_step = 1.e-2,
                    double print_step = -1.,
                    double max_step   = 1.);

    /**
     * Declare the control parameters for parameter handler.
     */
    static void
    declare_parameters(ParameterHandler & param);
    /**
     * Read the control parameters from a parameter handler.
     */
    void
    parse_parameters(ParameterHandler & param);

    /**
     * The left end of the time interval.
     */
    double
    start() const;
    /**
     * The right end of the time interval. The control mechanism ensures that
     * the final time step ends at this point.
     */
    double
    final() const;
    /**
     * The tolerance value controlling the time steps.
     */
    double
    tolerance() const;
    /**
     * The size of the current time step.
     */
    double
    step() const;

    /**
     * The current time.
     */
    double
    now() const;

    /**
     * Compute the size of the next step and return true if it differs from
     * the current step size. Advance the current time by the new step size.
     */
    bool
    advance();

    /**
     * Set start value.
     */
    void
    start(double);
    /**
     * Set final time value.
     */
    void
    final(double);
    /**
     * Set tolerance
     */
    void
    tolerance(double);
    /**
     * Set strategy.
     */
    void strategy(Strategy);

    /**
     * Set size of the first step. This may be overwritten by the time
     * stepping strategy.
     *
     * @param[in] step The size of the first step, which may be overwritten by
     * the time stepping strategy.
     */
    void
    start_step(const double step);

    /**
     * Set size of the maximum step size.
     */
    void
    max_step(double);

    /**
     * Set now() equal to start(). Initialize step() and print() to their
     * initial values.
     */
    void
    restart();
    /**
     * Return true if this timestep should be written to disk.
     */
    bool
    print();
    /**
     * Set the output name template.
     */
    void
    file_name_format(const char *);
    const char *
    file_name_format();

  private:
    double   start_val;
    double   final_val;
    double   tolerance_val;
    Strategy strategy_val;
    double   start_step_val;
    double   max_step_val;
    double   min_step_val;
    /**
     * The size of the current time step. This may differ from @p step_val, if
     * we aimed at @p final_val.
     */
    double current_step_val;
    double step_val;

    double now_val;
    double print_step;
    double next_print_val;

    char format[30];
  };

  inline double
  TimestepControl::start() const
  {
    return start_val;
  }

  inline double
  TimestepControl::final() const
  {
    return final_val;
  }

  inline double
  TimestepControl::step() const
  {
    return current_step_val;
  }

  inline double
  TimestepControl::tolerance() const
  {
    return tolerance_val;
  }

  inline double
  TimestepControl::now() const
  {
    return now_val;
  }

  inline void
  TimestepControl::start(double t)
  {
    start_val = t;
  }

  inline void
  TimestepControl::final(double t)
  {
    final_val = t;
  }

  inline void
  TimestepControl::tolerance(double t)
  {
    tolerance_val = t;
  }

  inline void
  TimestepControl::strategy(Strategy t)
  {
    strategy_val = t;
  }

  inline void
  TimestepControl::start_step(const double t)
  {
    start_step_val = t;
  }

  inline void
  TimestepControl::max_step(double t)
  {
    max_step_val = t;
  }

  inline void
  TimestepControl::restart()
  {
    now_val          = start_val;
    step_val         = start_step_val;
    current_step_val = step_val;
    if(print_step > 0.)
      next_print_val = now_val + print_step;
    else
      next_print_val = now_val - 1.;
  }

  inline void
  TimestepControl::file_name_format(const char * fmt)
  {
    strcpy(format, fmt);
  }

  inline const char *
  TimestepControl::file_name_format()
  {
    return format;
  }
} // namespace Algorithms

DEAL_II_NAMESPACE_CLOSE

#endif
