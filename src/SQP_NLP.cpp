#include "SQP_NLP.h"
#include <iostream>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <qpOASES.hpp>
#include <cstring>
#include <chrono>


SQP_NLP::SQP_NLP(VectorXreal w0_val, VectorXreal w_ref_val)
: w(w0_val), w_ref(w_ref_val),  w_size(w0_val.size()), flattened_f_grad(w_size), 
        flattened_B_k(2*w_size), g_size(1), flattened_g_Jac(w_size*g_size), flattened_g_eval(g_size)
    
    {
        previous_step.resize(w_size);
        if (g_size>0)
        {
            A.resize(g_size*w_size);
            lbA.resize(g_size);
            ubA.resize(g_size);
        }
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
    return 0.5* Residual(z,z_ref).squaredNorm();
}

VectorXreal SQP_NLP::g_constraints(const VectorXreal &z)
{
    VectorXreal r(1);
    r(0) = z(0) + pow((1-z(1)),2);

    return r;
}

VectorXreal SQP_NLP::Residual(const VectorXreal &z, const VectorXreal &z_ref)
{
    VectorXreal r(3);
    r(0) = z(0) - 1;              
    r(1) = 10.0 * (z(1) - z(0)*z(0));  
    r(2) = z(0);        
    return r; //w-w_ref;
}

void SQP_NLP::define_QP_params()
{
    auto Residual_bound = [this](VectorXreal& z, VectorXreal &z_ref) {
        return Residual(z, z_ref);
    };
    
    auto LS_cost_bound = [this](VectorXreal& z, VectorXreal &z_ref) {
        return LS_cost(z, z_ref);
    };

    auto g_constraints_bound = [this](VectorXreal &z) {
        return g_constraints(z);
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

    // Gauss-Newton Hessian approximation
    Eigen::MatrixXd B_k;
    B_k = R_Jac.transpose()*R_Jac;

    for (int i {0}; i<g_k.size() ; ++i)
    {
        g_k(i)*=-1;
    }

    fill_vector_array(w_size, flattened_f_grad.data(), LS_cost_grad);
    fill_matrix_array(2*w_size, flattened_B_k.data(), B_k);
    fill_matrix_array(g_size*w_size, flattened_g_Jac.data(), g_Jac.transpose());
    fill_vector_array(g_size, flattened_g_eval.data(), g_k);
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

    if (g_size>0)
    {
        std::memcpy(A.data(), flattened_g_Jac.data(), g_size * w_size * sizeof(qpOASES::real_t));
        std::memcpy(lbA.data(), flattened_g_eval.data(), g_size * sizeof(qpOASES::real_t));
        std::memcpy(ubA.data(), flattened_g_eval.data(), g_size * sizeof(qpOASES::real_t));

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


void SQP_NLP::solve(int max_qp_iter, double tol)
{
    
    qpOASES::QProblem QP(w_size,g_size);
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

            std::cout <<std::endl;
            std::cout << "#######################   NLP SOLUTION FOUND   #######################" << std::endl;
            std::cout <<std::endl;
            std::cout <<"               "<< "      SQP_ITER    |    ELAPSED TIME   "<< std::endl;
            std::cout <<"               "<< "------------------|-------------------"<< std::endl;
            std::cout <<"               "<< "         "<<i+1 << "        |    " << elapsed<< "    " << std::endl;
            std::cout <<std::endl;
            std::cout << "OPTIMAL SOLUTION W = (" << w.transpose() << " )" << std::endl;
            break;
        }
        
        w += previous_step;
    }
    

    
    
}

