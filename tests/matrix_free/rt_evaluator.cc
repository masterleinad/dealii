
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/matrix_free/evaluation_selector.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/polynomials_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <iostream>

using namespace dealii;

template <int dim, int fe_degree, int n_points, bool row_to_cols, typename Number>
void evaluate_rt(const AlignedVector<Number> &values_degree,
                 const AlignedVector<Number> &values_degree_plus_one,
                 const AlignedVector<Number> &input,
                 AlignedVector<Number> &output)
{
  AssertDimension(values_degree.size(), (fe_degree+1)*n_points);
  AssertDimension(values_degree_plus_one.size(), (fe_degree+2)*n_points);
  AssertDimension((dim*(fe_degree+2)*Utilities::fixed_int_power<fe_degree+1,dim-1>::value), input.size());
  AssertDimension((dim*Utilities::fixed_int_power<n_points,dim>::value), output.size());

  if (dim == 2)
    {
      // x component
      for (int i=fe_degree; i>=0; --i)
        internal::EvaluatorTensorProduct<internal::evaluate_general,1,fe_degree+2,n_points,Number>
            ::template apply<0,true,false>(values_degree_plus_one.begin(),
                                                 input.begin()+i*(fe_degree+2),
                                                 output.begin()+i*n_points);
      internal::EvaluatorTensorProduct<internal::evaluate_general,2,fe_degree+1,n_points,Number>
          ::template apply<1,true,false>(values_degree.begin(),
                                               output.begin(),
                                               output.begin());

      // y component
      for (int i=fe_degree+1; i>=0; --i)
        internal::EvaluatorTensorProduct<internal::evaluate_general,1,fe_degree+1,n_points,Number>
            ::template apply<0,true,false>(values_degree.begin(),
                                                 input.begin()+(fe_degree+1)*(fe_degree+2)+i*(fe_degree+1),
                                                 output.begin()+n_points*n_points+i*n_points);
      internal::EvaluatorTensorProduct<internal::evaluate_general,2,fe_degree+2,n_points,Number>
          ::template apply<1,true,false>(values_degree_plus_one.begin(),
                                               output.begin()+n_points*n_points,
                                               output.begin()+n_points*n_points);
    }
  else if (dim == 3)
    {
      // x component
      for (int z=0; z<fe_degree+1; ++z)
        {
          for (int i=fe_degree; i>=0; --i)
            internal::EvaluatorTensorProduct<internal::evaluate_general,1,fe_degree+2,n_points,Number>
                ::template apply<0,true,false>(values_degree_plus_one.begin(),
                                                     input.begin()+i*(fe_degree+2)+z*(fe_degree+1)*(fe_degree+2),
                                                     output.begin()+i*n_points+z*n_points*n_points);
          // y direction
          internal::EvaluatorTensorProduct<internal::evaluate_general,2,fe_degree+1,n_points,Number>
              ::template apply<1,true,false>(values_degree.begin(),
                                                   output.begin()+z*n_points*n_points,
                                                   output.begin()+z*n_points*n_points);
        }
      internal::EvaluatorTensorProduct<internal::evaluate_general,3,fe_degree+1,n_points,Number>
          ::template apply<2,true,false>(values_degree.begin(),
                                               output.begin(),
                                               output.begin());

      // y component
      for (int z=0; z<fe_degree+1; ++z)
        {
          for (int i=fe_degree+1; i>=0; --i)
            internal::EvaluatorTensorProduct<internal::evaluate_general,1,fe_degree+1,n_points,Number>
                ::template apply<0,true,false>(values_degree.begin(),
                                                     input.begin()+(z+fe_degree+1)*(fe_degree+1)*(fe_degree+2)+
                                                                   i*(fe_degree+1),
                                                     output.begin()+(z+n_points)*n_points*n_points+i*n_points);
          internal::EvaluatorTensorProduct<internal::evaluate_general,2,fe_degree+2,n_points,Number>
              ::template apply<1,true,false>(values_degree_plus_one.begin(),
                                                   output.begin()+(n_points+z)*n_points*n_points,
                                                   output.begin()+(n_points+z)*n_points*n_points);
        }
      internal::EvaluatorTensorProduct<internal::evaluate_general,3,fe_degree+1,n_points,Number>
          ::template apply<2,true,false>(values_degree.begin(),
                                               output.begin()+n_points*n_points*n_points,
                                               output.begin()+n_points*n_points*n_points);

      // z component
      for (int z=0; z<fe_degree+2; ++z)
        {
          for (int i=fe_degree+1; i>=0; --i)
            internal::EvaluatorTensorProduct<internal::evaluate_general,1,fe_degree+1,n_points,Number>
                ::template apply<0,true,false>(values_degree.begin(),
                                                     input.begin()+(z+2*(fe_degree+1))*(fe_degree+1)*(fe_degree+2)+
                                                                   i*(fe_degree+1),
                                                     output.begin()+(z+2*n_points)*n_points*n_points+i*n_points);
          internal::EvaluatorTensorProduct<internal::evaluate_general,2,fe_degree+1,n_points,Number>
              ::template apply<1,true,false>(values_degree.begin(),
                                                   output.begin()+(2*n_points+z)*n_points*n_points,
                                                   output.begin()+(2*n_points+z)*n_points*n_points);
        }
      internal::EvaluatorTensorProduct<internal::evaluate_general,3,fe_degree+2,n_points,Number>
          ::template apply<2,true,false>(values_degree_plus_one.begin(),
                                               output.begin()+2*n_points*n_points*n_points,
                                               output.begin()+2*n_points*n_points*n_points);
    }
}


template <int dim, int fe_degree, int n_points>
void test()
{
  FE_RaviartThomasNodal<dim> fe(fe_degree);
  QGauss<1> gauss_1d(n_points);
  Quadrature<dim> quadrature(gauss_1d);

  AlignedVector<double> values_in(fe.dofs_per_cell), values_out(quadrature.size()*dim), values_ref(quadrature.size()*dim);

  for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
    values_in[i] = (double)rand()/RAND_MAX;

  std::vector<unsigned int> lexicographic(fe.dofs_per_cell);

  QIterated<dim> output_quad(QTrapez<1>(), 8);
  for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
    {
      std::cout << "dof index " << i << std::endl;

    for (unsigned int d=0; d<dim; ++d)
      {
        for (unsigned int q=0; q<output_quad.size(); ++q)
          std::cout << fe.shape_value_component(i, output_quad.point(q), d) << " ";
        std::cout << std::endl;
      }
      }

  for (unsigned int i=0; i<(fe_degree+1)*(fe_degree+2); ++i)
    lexicographic[i] = i;
  for (unsigned int i=0; i<fe_degree+1; ++i)
    for (unsigned int j=0; j<fe_degree+2; ++j)
      lexicographic[(fe_degree+1)*(fe_degree+2)+i+j*(fe_degree+1)] = (fe_degree+1)*(fe_degree+2)+j+i*(fe_degree+2);

  PolynomialsRaviartThomas<dim> poly(fe_degree);

  std::vector<Tensor<1,dim> > values(dim*(fe_degree+2)*Utilities::fixed_int_power<fe_degree+1,dim-1>::value);
  std::vector<Tensor<2,dim>> grads;
  std::vector<Tensor<3,dim>> second;
  std::vector<Tensor<4,dim>> third;
  std::vector<Tensor<5,dim>> fourth;
  for (unsigned int q=0; q<quadrature.size(); ++q)
    {
      poly.compute(quadrature.point(q), values, grads, second, third, fourth);
      Tensor<1,dim> sum;
//      std::cout << "shape values: ";
//      for (unsigned int i=0; i<values.size(); ++i)
//        std::cout << values[i] << "    ";
//      std::cout << std::endl;
      for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
        {
          for (unsigned int d=0; d<dim; ++d)
            sum[d] += values[lexicographic[i]][d] * values_in[i];
        }
      for (unsigned int d=0; d<dim; ++d)
        values_ref[q+d*quadrature.size()] = sum[d];
      for (unsigned int d=0; d<dim; ++d)
        std::cout << sum[d] << " ";
      std::cout << std::endl;
    }
  AlignedVector<double> shape_values_degree((fe_degree+1)*n_points);
  AlignedVector<double> shape_values_degree_plus_one((fe_degree+2)*n_points);
  {
    FE_DGQ<1> fe1(fe_degree+1);
    FE_DGQ<1> fe2(fe_degree);
    for (unsigned int i=0; i<fe2.dofs_per_cell; ++i)
      for (unsigned int q=0; q<gauss_1d.size(); ++q)
        shape_values_degree[i*gauss_1d.size()+q] = fe2.shape_value(i, gauss_1d.point(q));
    for (unsigned int i=0; i<fe1.dofs_per_cell; ++i)
      for (unsigned int q=0; q<gauss_1d.size(); ++q)
        shape_values_degree_plus_one[i*gauss_1d.size()+q] = fe1.shape_value(i, gauss_1d.point(q));
  }

  evaluate_rt<dim,fe_degree,n_points,true,double>(shape_values_degree, shape_values_degree_plus_one,
                                                  values_in, values_out);

  internal::FEEvaluationImplAni<internal::MatrixFreeFunctions::tensor_general, FE_RaviartThomas<dim>, dim,
            fe_degree, fe_degree+1, double,0> fe_evaluation;

  //apply_anisotropic

  std::cout << "Input values: ";
  for (unsigned int i=0; i<values_in.size(); ++i)
    std::cout << values_in[i] << " ";
  std::cout << std::endl;

  std::cout << "Computed values: ";
  for (unsigned q=0; q<values_out.size(); ++q)
    std::cout << values_out[q] << " ";
  std::cout << std::endl;
  std::cout << "Errors: ";
  for (unsigned q=0; q<values_out.size(); ++q)
    std::cout << values_out[q] - values_ref[q] << " ";
  std::cout << std::endl;
}


int main()
{
  test<2,1,3>();
  //test<2,2,4>();
  //test<2,2,5>();
}

