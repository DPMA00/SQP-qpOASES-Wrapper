#include "SQP_NLP.h"
#include <iostream>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <qpOASES.hpp>
#include <cstring>
#include <chrono>


void SQP_NLP::fill_vector_array(size_t length, qpOASES::real_t *array, const Eigen::VectorXd &vector)
{
    for (int i{0}; i < length; ++i)
    {
        array[i] = vector(i);
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

VectorXreal SQP_NLP::Residual(const VectorXreal &z, const VectorXreal &z_ref)
{
    VectorXreal r(2);
    r(0) = 5 - z(0);                  // First residual term
    r(1) = sqrt(3) * (z(1) - z(0)*z(0));
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

    // Gauss-Newton Hessian approximation
    Eigen::MatrixXd B_k;
    B_k = R_Jac.transpose()*R_Jac;


    fill_vector_array(w_size, flattened_f_grad.data(), LS_cost_grad);
    fill_matrix_array(2*w_size, flattened_B_k.data(), B_k);
}


void SQP_NLP::solve_QP_iter()
{
    define_QP_params();

    qpOASES::QProblem QP(w_size,0);

    qpOASES::real_t H[2*w_size];
    std::memcpy(H, flattened_B_k.data(), 2*w_size * sizeof(qpOASES::real_t));


    qpOASES::real_t g[w_size];
    std::memcpy(g, flattened_f_grad.data(), w_size * sizeof(qpOASES::real_t));  
    
    qpOASES::real_t lb[2] = {-1e5, -1e5};
    qpOASES::real_t ub[2] = { 1e5,  1e5};

    qpOASES::real_t* A = nullptr;
    qpOASES::real_t* lbA = nullptr;
    qpOASES::real_t* ubA = nullptr;

    int nWSR = 10;

    qpOASES::Options options;
    options.printLevel = qpOASES::PL_NONE;
    QP.setOptions(options);
    
    qpOASES::returnValue status = QP.init(H, g, A, lb, ub, lbA, ubA, nWSR);
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
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i{0}; i<max_qp_iter; ++i)
    {
        std::cout << "Starting iteration " << i << std::endl;
        solve_QP_iter();
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

