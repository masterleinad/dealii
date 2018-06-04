// ---------------------------------------------------------------------
//
// Copyright (C) 2018 by the deal.II authors
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

#ifndef dealii_polynomials_hermite_h
#define dealii_polynomials_hermite_h

#include <deal.II/base/polynomial.h>

#include <fstream>
#include <iostream>

DEAL_II_NAMESPACE_OPEN

/**
 * This class implements Hermite basis polynomials
 * fulfilling the following properties:
 * - $p(x_0)=y_0$
 * - $p'(x_0)=y'_0$
 * - $p(x_1)=y_1$
 * - $p(x_2)=y_2$
 * - ...
 * - $p(x_n)=y_n$
 * - $p'(x_n)=y'_n$
 *
 * In particular, the resulting polynomial is given by the values and
 * derivatives in the end points and the values at the support points
 * inside the end points. Therefore, the minimal degree is three.
 *
 * @ingroup Polynomials
 * @author Daniel Arndt
 * @date 2018
 */
template <typename number>
class PolynomialsHermite : public Polynomials::Polynomial<number>
{
public:
  /**
   * Construct the @p index-th polynomial with respect to the @p support_points.
   * - The first polynomial is one in the first support point
   *   and has zero derivative there. In all the other support
   *   points the polynomials is zero.
   * - The second polynomial is zero in all support points
   *   and its derivative is one in the first support point.
   * - The third polynomial is one in the last support point
   *   and has zero derivative there. In all the other support
   *   points the polynomial is zero.
   * - The forth polynomial is zero in all support points
   *   and its derivative is one in the last support point.
   * - The remaining polynomials is one in one support point and zero in all
   *   the others. The derivative at the first and last support point is zero.
   */
  PolynomialsHermite(const std::vector<number> &support_points,
                     const unsigned int         index)
  {
    const unsigned int n_support_points = support_points.size();
    const unsigned int degree           = n_support_points + 1;
    Assert(n_support_points >= 2,
           ExcMessage("This class only makes sense for degree>=3!"));
    AssertIndexRange(index, n_support_points + 2);

    std::vector<number> actual_support_points;
    actual_support_points.reserve(n_support_points + 2);
    actual_support_points.push_back(support_points[0]);
    actual_support_points.insert(actual_support_points.end(),
                                 support_points.begin(),
                                 support_points.end());
    actual_support_points.push_back(support_points[n_support_points - 1]);

    int transposed_index = index;
    if (index == 2 || index == 3)
      transposed_index = index + degree - 3;
    else if (index > 3)
      transposed_index = index - 2;

    std::vector<number> values(degree + 1, 0.);
    values[transposed_index] = 1.;

    this->coefficients = compute_monomial_coefficients(
      compute_newton_coefficients(actual_support_points, values),
      actual_support_points);
  }

  /**
   * Same as above but use equidistant support points such that the total degree
   * is @p degree (which must me greater or equal three).
   */
  PolynomialsHermite(const unsigned int degree, const unsigned int index)
  {
    Assert(degree >= 3,
           ExcMessage("This class only makes sense for degree>=3!"));
    AssertIndexRange(index, degree + 1);
    std::vector<number> support_points(degree + 1, 0.);
    support_points[degree - 1] = 1.;
    support_points[degree]     = 1.;
    for (unsigned int i = 2; i < degree - 1; ++i)
      support_points[i] = (i - 1) * 1. / (degree - 2);

    int transposed_index = index;
    if (index == 2 || index == 3)
      transposed_index = index + degree - 3;
    else if (index > 3)
      transposed_index = index - 2;

    std::vector<number> values(degree + 1, 0.);
    values[transposed_index] = 1.;

    this->coefficients = compute_monomial_coefficients(
      compute_newton_coefficients(support_points, values), support_points);
  }

  /**
   * Build a complete polynomial basis using the given @p support_points.
   */
  static std::vector<Polynomials::Polynomial<number>>
  generate_complete_basis(const std::vector<number> &support_points)
  {
    const unsigned int n_support_points = support_points.size();
    const unsigned int degree           = n_support_points + 1;
    Assert(n_support_points >= 2,
           ExcMessage("This class only makes sense for degree>=3!"));
    std::vector<number> actual_support_points;
    actual_support_points.reserve(n_support_points + 2);
    actual_support_points.push_back(support_points[0]);
    actual_support_points.insert(actual_support_points.end(),
                                 support_points.begin(),
                                 support_points.end());
    actual_support_points.push_back(support_points[n_support_points - 1]);

    std::vector<Polynomials::Polynomial<number>> all_coefficients(
      n_support_points + 2);
    for (unsigned int index = 0; index < n_support_points + 2; ++index)
      {
        int transposed_index = index;
        if (index == 2 || index == 3)
          transposed_index = index + degree - 3;
        else if (index > 3)
          transposed_index = index - 2;

        std::vector<number> values(degree + 1, 0.);
        values[transposed_index] = 1.;

        all_coefficients[index] =
          Polynomials::Polynomial<number>(compute_monomial_coefficients(
            compute_newton_coefficients(actual_support_points, values),
            actual_support_points));
      }
    return all_coefficients;
  }

  /**
   * Build a complete polynomial basis of the given @p degree. The support
   * points are equidistant and the degree must at least be three.
   */
  static std::vector<Polynomials::Polynomial<number>>
  generate_complete_basis(const unsigned int degree)
  {
    Assert(degree >= 3,
           ExcMessage("This class only makes sense for degree>=3!"));
    std::vector<number> support_points(degree + 1, 0.);
    support_points[degree - 1] = 1.;
    support_points[degree]     = 1.;
    for (unsigned int i = 2; i < degree - 1; ++i)
      support_points[i] = (i - 1) * 1. / (degree - 2);

    std::vector<Polynomials::Polynomial<number>> all_coefficients(degree + 1);
    for (unsigned int index = 0; index < degree + 1; ++index)
      {
        int transposed_index = index;
        if (index == 2 || index == 3)
          transposed_index = index + degree - 3;
        else if (index > 3)
          transposed_index = index - 2;

        std::vector<number> values(degree + 1, 0.);
        values[transposed_index] = 1.;

        all_coefficients[index] =
          Polynomials::Polynomial<number>(compute_monomial_coefficients(
            compute_newton_coefficients(support_points, values),
            support_points));
      }
    return all_coefficients;
  };

  /**
   * Compute the coefficients of the Newton polynomial that has values @p y in
   * the support points @p x which are assumed to be ordered.
   * Repeated support points indicate that higher derivatives are also used for
   * interpolation, i.e. choosing the values as follows
   * - x[0] = 0., x[1] = 0., x[2] = 1., x[3] = 1.
   * - y[0] = 1., y[1] = 2., y[2] = 3., y[3] = 4.
   *
   * results in a polynomial with value 1 and derivative 2 at 0. and
   * value 3 and derivative 4 at 1.
   */
  static std::vector<number>
  compute_newton_coefficients(const std::vector<number> &x,
                              const std::vector<number> &y)
  {
    const unsigned int n = x.size();
    Assert(x.size() == y.size(),
           ExcMessage("The number of support points and the number of values "
                      "must be the same"));
    Assert(std::is_sorted(x.begin(), x.end()),
           ExcMessage("The std::vector of support points must be sorted!"));

    std::vector<number> coeffs(n);

    coeffs[0]               = y[0];
    unsigned int firstIndex = 0;
    for (unsigned int i = 1; i < n; ++i)
      {
        if (std::abs(x[i - 1] - x[i]) > 1.e-12)
          firstIndex = i;
        coeffs[i] = y[firstIndex];
      }

    for (unsigned int i = 1; i < n; ++i)
      for (unsigned int j = n - 1; j >= i; --j)
        {
          if (std::abs(x[j] - x[j - i]) > 1.e-12)
            coeffs[j] = (coeffs[j] - coeffs[j - 1]) / (x[j] - x[j - i]);
          else
            {
              // Find the first index that has the same x-value
              int k = j;
              while (std::abs(x[k] - x[k - 1]) < 1.e-12 && k > 0)
                --k;
              // Assign the ith derivative
              int factor = 1;
              for (unsigned int w = 2; w <= i; ++w)
                factor *= w;
              coeffs[j] = y[k + i] / factor;
            }
        }

    return coeffs;
  }

  /**
   * Compute from the given Newton polynomial coefficients @p coeffs
   * (with respect to @p support_points) the respective coefficients
   * for a monomial basis.
   */
  static std::vector<number>
  compute_monomial_coefficients(const std::vector<number> &newton_coefficients,
                                const std::vector<number> &support_points)
  {
    const unsigned int  n = newton_coefficients.size();
    std::vector<number> monomials(n);
    monomials[0] = newton_coefficients[n - 1];
    for (unsigned int i = 1; i < n; ++i)
      {
        for (unsigned int j = i; j > 0; --j)
          monomials[j] =
            monomials[j - 1] - support_points[n - 1 - i] * monomials[j];
        monomials[0] = newton_coefficients[n - 1 - i] -
                       support_points[n - 1 - i] * monomials[0];
      }
    return monomials;
  }
};

DEAL_II_NAMESPACE_CLOSE

#endif
