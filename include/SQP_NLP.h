#ifndef _SQP_NLP_
#define _SQP_NLP_

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <qpOASES.hpp>
#include <functional>

using VectorXreal = autodiff::VectorXreal;
using real = autodiff::real;
using ResidualFunc = std::function<VectorXreal(const VectorXreal&, const VectorXreal&)>;
using EqualityFunc = std::function<VectorXreal(const VectorXreal&)>;
using InequalityFunc = std::function<VectorXreal(const VectorXreal&)>;


class SQP_NLP{
public:
    SQP_NLP(int N, int nx, int nu, int ng, int nh);
    
    Eigen::VectorXd get_solution();
    
    void set_initial_guess(const VectorXreal w0_val);
    void set_reference(const VectorXreal w_ref_val);
    void set_x_bounds(VectorXreal lbx_value, VectorXreal ubx_value);
    void set_u_bounds(VectorXreal lbu_value, VectorXreal ubu_value);
    void set_h_bounds(VectorXreal lbh_value, VectorXreal ubh_value);

    void set_residual_expr(ResidualFunc f);
    void set_g_expr(EqualityFunc g);
    void set_h_expr(InequalityFunc h);
    
    
    void solve(int max_qp_iter=100, double tol=1e-6, int print_level = 1);
    
    
    

private:
    VectorXreal w;
    VectorXreal previous_step;
    VectorXreal w_ref;
    VectorXreal lbh, ubh;
    ResidualFunc residual;
    EqualityFunc g_constr;
    InequalityFunc h_constr;

    const VectorXreal lbx, ubx, lbu, ubu;
    int w_size, g_size, h_size, gh_size;
       
    bool solved_ = false;


    std::vector<qpOASES::real_t> flattened_f_grad;
    std::vector<qpOASES::real_t> flattened_B_k;
    std::vector<qpOASES::real_t> flattened_g_Jac;
    std::vector<qpOASES::real_t> flattened_g_eval;
    std::vector<qpOASES::real_t> flattened_h_Jac;
    std::vector<qpOASES::real_t> flattened_h_eval;
    std::vector<qpOASES::real_t> A, lbA, ubA;
    std::vector<qpOASES::real_t> lbx_, ubx_, lbu_, ubu_;
    std::vector<qpOASES::real_t> flattened_constraint_Jac;
    std::vector<qpOASES::real_t> flattened_constraint_lb_eval;
    std::vector<qpOASES::real_t> flattened_constraint_ub_eval;
    


    VectorXreal set_lbh(const VectorXreal &h_k);
    VectorXreal set_ubh(const VectorXreal &h_k);
    void fill_vector_array(size_t length, qpOASES::real_t *array, const Eigen::VectorXd &vector);
    void fill_matrix_array(size_t length, qpOASES::real_t *array, const Eigen::MatrixXd &matrix);
    void fill_vector_array(size_t length, qpOASES::real_t *array, const VectorXreal &vector);

    real LS_cost(const VectorXreal &z, const VectorXreal &z_ref);
    void define_QP_params();
    void solve_QP_iter(qpOASES::QProblem QP);
    void print_table(int iter, double elapsed_sec);


};

#endif