#ifndef _SQP_NLP_
#define _SQP_NLP_

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <qpOASES.hpp>


using VectorXreal = autodiff::VectorXreal;
using real = autodiff::real;

class SQP_NLP{
    int w_size;
       
public:
    SQP_NLP(VectorXreal w0_val, VectorXreal w_ref_val)
    : w(w0_val), w_ref(w_ref_val),  w_size(w0_val.size()), flattened_f_grad(w_size), flattened_B_k(2*w_size)
    {
        previous_step.resize(w_size);
    }

    VectorXreal Residual(const VectorXreal &z, const VectorXreal &z_ref);
    void solve(int max_qp_iter=100, double tol=1e-6);

    

private:
    VectorXreal w;
    VectorXreal previous_step;
    const VectorXreal w_ref;

    std::vector<qpOASES::real_t> flattened_f_grad;
    std::vector<qpOASES::real_t> flattened_B_k;

    void fill_vector_array(size_t length, qpOASES::real_t *array, const Eigen::VectorXd &vector);
    void fill_matrix_array(size_t length, qpOASES::real_t *array, const Eigen::MatrixXd &matrix);
    real LS_cost(const VectorXreal &z, const VectorXreal &z_ref);
    void define_QP_params();
    void solve_QP_iter();

};

#endif