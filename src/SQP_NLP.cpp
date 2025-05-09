#include "SQP_NLP.h"
#include <iostream>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <qpOASES.hpp>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <sstream>

SQP_NLP::SQP_NLP(int N, int nx, int nu, int ng, int nh)
    : w_size {(nx+nu)*N + nx}
    , flattened_f_grad(w_size)
    , flattened_B_k(2*w_size)
    , g_size(ng)
    , h_size(nh)
    , flattened_g_Jac(w_size*g_size)
    , flattened_g_eval(g_size)
    , flattened_h_Jac(w_size*h_size)
    , flattened_h_eval(h_size)
    , gh_size(g_size+h_size)
    , flattened_constraint_Jac(0)
    , flattened_constraint_lb_eval(0)
    , flattened_constraint_ub_eval(0)
    , lbx_(0)
    , ubx_(0)
    , lbu_(0)
    , ubu_(0)
    
    {
        previous_step.resize(w_size);
        if (gh_size>0)
        {
            A.resize(gh_size*w_size);
            lbA.resize(gh_size);
            ubA.resize(gh_size);
        }
    }

void SQP_NLP::set_initial_guess(const VectorXreal w0_val)
{
    w=w0_val;
}


void SQP_NLP::set_reference(const VectorXreal w_ref_val)
{
    if (w_ref_val.size() != w_size)
    {
        std::cerr << "Incompatible dimensions: \"w\"=" << w_size << ", " << "\"w_ref\"" << w_ref_val.size() << "\n";
        std::exit(EXIT_FAILURE);
    }
    else
    {
        w_ref = w_ref_val;
    }
}


void SQP_NLP::set_x_bounds(VectorXreal lbx_value, VectorXreal ubx_value)
{
    lbx(lbx_value);
    ubx(ubx_value);
}
void SQP_NLP::set_u_bounds(VectorXreal lbu_value, VectorXreal ubu_value)
{
    lbu(lbu_value);
    ubu(ubu_value);

}
void SQP_NLP::set_h_bounds(VectorXreal lbh_value, VectorXreal ubh_value)
{
    lbh=lbh_value;
    ubh=ubh_value;
}

void SQP_NLP::print_table(const int iter, const double elapsed_sec)
{
    std::ostringstream w_stream;
    w_stream << std::fixed << std::setprecision(6) << w.transpose();
    std::string w_str = w_stream.str();

    const int col1 {20};
    const int col2 {20};
    const int total_width = col1+col2;

    std::cout << std::string((total_width-30)/2, '#')
              << "  NLP SOLUTION FOUND  "
              << std::string((total_width-30)/2, '#') << "\n\n";

    std::cout << std::left
              << std::setw(col1) << "Nr SQP Iterations"
              << std::setw(col2) << "Elapsed Time (s)"  << "\n";

    std::cout << std::string(total_width, '-') << "\n";

    std::cout << std::left
              << std::setw(col1) << iter
              << std::setw(col2) << std::fixed << std::setprecision(9) << elapsed_sec  << "\n";


}

void SQP_NLP::fill_vector_array(size_t length, qpOASES::real_t *array, const Eigen::VectorXd &vector)
{
    for (int i{0}; i < length; ++i)
    {
        array[i] = vector(i);
    }
}
void SQP_NLP::fill_vector_array(size_t length, qpOASES::real_t *array, const VectorXreal &vector)
{
    for (int i{0}; i < length; ++i)
    {
        array[i] = val(vector(i));
    }
}

void SQP_NLP::fill_matrix_array(size_t length, qpOASES::real_t *array, const Eigen::MatrixXd &matrix)
{
    Eigen::MatrixXd intermediate = matrix.reshaped();
    for (int i{0}; i < length; ++i)
    {
        array[i] = intermediate(i);
    }
}




real SQP_NLP::LS_cost(const VectorXreal &z, const VectorXreal &z_ref)
{
    return 0.5* residual(z,z_ref).squaredNorm();
}

void SQP_NLP::set_residual_expr(ResidualFunc f)
{
    residual = f;
}
void SQP_NLP::set_g_expr(EqualityFunc g)
{
    g_constr = g;
}
void SQP_NLP::set_h_expr(InequalityFunc h)
{
    h_constr = h;
}

VectorXreal SQP_NLP::set_lbh(const VectorXreal &h_k)
{
    return -h_k + lbh;
    
}

VectorXreal SQP_NLP::set_ubh(const VectorXreal &h_k)
{
    return -h_k + ubh;

}

void SQP_NLP::define_QP_params()
{
    flattened_constraint_lb_eval.clear();
    flattened_constraint_ub_eval.clear();
    flattened_constraint_Jac.clear();
    
    auto Residual_bound = [this](VectorXreal& z, VectorXreal &z_ref) {
        return residual(z, z_ref);
    };
    
    auto LS_cost_bound = [this](VectorXreal& z, VectorXreal &z_ref) {
        return LS_cost(z, z_ref);
    };

    auto g_constraints_bound = [this](VectorXreal &z) {
        return g_constr(z);
    };

    auto h_constraints_bound = [this](VectorXreal &z) {
        return h_constr(z);
    };

    VectorXreal z = w;
    VectorXreal z_ref = w_ref;
    
    // Gradient of the LS Objective
    real LS_cost_eval;                                     
    Eigen::VectorXd LS_cost_grad;
    gradient(LS_cost_bound, wrt(z), at(z, z_ref), LS_cost_eval, LS_cost_grad);
    
    // Jacobian of the Residual R(w)
    Eigen::MatrixXd R_Jac;
    VectorXreal Res_eval;
    jacobian(Residual_bound, wrt(z), at(z, z_ref), Res_eval, R_Jac);
    
    // Jacobian of the equality constraints
    VectorXreal g_k;
    Eigen::MatrixXd g_Jac;

    jacobian(g_constraints_bound, wrt(z), at(z), g_k, g_Jac);

    for (int i {0}; i<g_k.size() ; ++i)
    {
        g_k(i)*=-1;
    }
    
    // Jacobian of the inequality constraints
    VectorXreal h_k;
    Eigen::MatrixXd h_Jac;
    jacobian(h_constraints_bound, wrt(z), at(z), h_k, h_Jac);
    
    // Gauss-Newton Hessian approximation
    Eigen::MatrixXd B_k;
    B_k = R_Jac.transpose()*R_Jac;
    
    VectorXreal lbh_QP = set_lbh(h_k);
    VectorXreal ubh_QP = set_ubh(h_k);
    
   
    fill_vector_array(w_size, flattened_f_grad.data(), LS_cost_grad);
    
    fill_matrix_array(2*w_size, flattened_B_k.data(), B_k);
    fill_matrix_array(g_size*w_size, flattened_g_Jac.data(), g_Jac.transpose());
    fill_vector_array(g_size, flattened_g_eval.data(), g_k);
    fill_matrix_array(h_size*w_size, flattened_h_Jac.data(), h_Jac.transpose());


    fill_vector_array(h_size, flattened_h_eval.data(), lbh_QP);

    flattened_constraint_lb_eval.insert(flattened_constraint_lb_eval.end(),flattened_g_eval.begin(), flattened_g_eval.end());
    flattened_constraint_ub_eval = flattened_constraint_lb_eval;
    flattened_constraint_lb_eval.insert(flattened_constraint_lb_eval.end(),flattened_h_eval.begin(), flattened_h_eval.end());

    fill_vector_array(h_size, flattened_h_eval.data(), ubh_QP);
    flattened_constraint_ub_eval.insert(flattened_constraint_ub_eval.end(),flattened_h_eval.begin(), flattened_h_eval.end());

    flattened_constraint_Jac.insert(flattened_constraint_Jac.end(),flattened_g_Jac.begin(), flattened_g_Jac.end());
    flattened_constraint_Jac.insert(flattened_constraint_Jac.end(),flattened_h_Jac.begin(), flattened_h_Jac.end());

    

}


void SQP_NLP::solve_QP_iter(qpOASES::QProblem QP)
{
    define_QP_params();

    qpOASES::real_t H[2*w_size];
    std::memcpy(H, flattened_B_k.data(), 2*w_size * sizeof(qpOASES::real_t));


    qpOASES::real_t g[w_size];
    std::memcpy(g, flattened_f_grad.data(), w_size * sizeof(qpOASES::real_t));  
    
    qpOASES::real_t lb[2] = {-1e5, -1e5};
    qpOASES::real_t ub[2] = { 1e5,  1e5};


    qpOASES::real_t* A_ptr = nullptr;
    qpOASES::real_t* lbA_ptr = nullptr;
    qpOASES::real_t* ubA_ptr = nullptr;

    if (gh_size>0)
    {
        std::memcpy(A.data(), flattened_constraint_Jac.data(), gh_size * w_size * sizeof(qpOASES::real_t));
        std::memcpy(lbA.data(), flattened_constraint_lb_eval.data(), gh_size * sizeof(qpOASES::real_t));
        std::memcpy(ubA.data(), flattened_constraint_ub_eval.data(), gh_size * sizeof(qpOASES::real_t));

        A_ptr   = A.data();
        lbA_ptr = lbA.data();
        ubA_ptr = ubA.data();
    }

    

    

    
    int nWSR = 10;
    qpOASES::returnValue status = QP.init(H, g, A_ptr, lb, ub, lbA_ptr, ubA_ptr, nWSR);
    if (status != qpOASES::SUCCESSFUL_RETURN) {
        std::cerr << "QP.init failed with code: " << status << std::endl;
    }

    qpOASES::real_t xOpt [w_size];
    QP.getPrimalSolution(xOpt);

    previous_step(0) = xOpt[0];
    previous_step(1) = xOpt[1];
}


void SQP_NLP::solve(int max_qp_iter, double tol, int print_level)
{
    
    qpOASES::QProblem QP(w_size,gh_size);
    qpOASES::Options options;
    options.printLevel = qpOASES::PL_NONE;
    QP.setOptions(options);
    

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i{0}; i<max_qp_iter; ++i)
    {
        solve_QP_iter(QP);
        if (previous_step.norm() < tol)
        {
            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = t2-t1;
            solved_ = true;
            if (print_level ==1)
            {
                print_table(i+1, elapsed.count());
            }         
            break;
        }
        
        w += previous_step;
    }    
    
}

Eigen::VectorXd SQP_NLP::get_solution()
{
    if (!solved_)
    {
        throw std::logic_error("Solution not available: call solve() first.");
    }
    return w.cast<double>();
}
