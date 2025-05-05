#ifndef _SQP_NLP_
#define _SQP_NLP_

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <qpOASES.hpp>


using VectorXreal = autodiff::VectorXreal;
using real = autodiff::real;

class SQP_NLP{
    int w_size;
    int g_size;
    int h_size;
    int gh_size;
       
public:
    SQP_NLP(VectorXreal w0_val, VectorXreal w_ref_val, VectorXreal lbh_value, VectorXreal ubh_value);

    VectorXreal Residual(const VectorXreal &z, const VectorXreal &z_ref);

    VectorXreal g_constraints(const VectorXreal &z);
    VectorXreal h_constraints(const VectorXreal &z);
    VectorXreal set_lbh(const VectorXreal &h_k);
    VectorXreal set_ubh(const VectorXreal &h_k);
    Eigen::VectorXd get_solution();
    
    void solve(int max_qp_iter=100, double tol=1e-6, int print_level = 1);
    
    
    

private:
    VectorXreal w;
    VectorXreal previous_step;
    const VectorXreal w_ref;
    VectorXreal lbh;
    VectorXreal ubh;
    bool solved_ = false;


    std::vector<qpOASES::real_t> flattened_f_grad;
    std::vector<qpOASES::real_t> flattened_B_k;
    std::vector<qpOASES::real_t> flattened_g_Jac;
    std::vector<qpOASES::real_t> flattened_g_eval;
    std::vector<qpOASES::real_t> flattened_h_Jac;
    std::vector<qpOASES::real_t> flattened_h_eval;
    std::vector<qpOASES::real_t> A, lbA, ubA;
    std::vector<qpOASES::real_t> flattened_constraint_Jac;
    std::vector<qpOASES::real_t> flattened_constraint_lb_eval;
    std::vector<qpOASES::real_t> flattened_constraint_ub_eval;
    


    
    void fill_vector_array(size_t length, qpOASES::real_t *array, const Eigen::VectorXd &vector);
    void fill_matrix_array(size_t length, qpOASES::real_t *array, const Eigen::MatrixXd &matrix);
    void fill_vector_array(size_t length, qpOASES::real_t *array, const VectorXreal &vector);

    real LS_cost(const VectorXreal &z, const VectorXreal &z_ref);
    void define_QP_params();
    void solve_QP_iter(qpOASES::QProblem QP);
    void print_table(int iter, double elapsed_sec);


};

#endif