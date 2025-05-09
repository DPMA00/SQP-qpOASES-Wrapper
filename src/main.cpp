#include "SQP_NLP.h"



VectorXreal g_constraints(const VectorXreal &z)
{
    VectorXreal r(1);
    r(0) = sin(z(0))-pow(z(1),2);

    return r;
}

VectorXreal h_constraints(const VectorXreal &z)
{
    VectorXreal r(1);
    r(0) = pow(z(0),2) + pow(z(1),2) - 4;

    return r;
}


VectorXreal residual(const VectorXreal &z, const VectorXreal &z_ref)
{
    VectorXreal r(2);
    r(0) = z(0)-4;
    r(1) = z(1)-4;
    return r;
}

int main()
{
    
    VectorXreal x(2);
    x << 2,-4;

    VectorXreal x_ref(2);
    x_ref << 0,0;

    VectorXreal lbh(1);
    lbh << -1e5;

    VectorXreal ubh(1);
    ubh << 0;

    VectorXreal lbx(0);
    
    VectorXreal ubx(0);

    VectorXreal lbu(0);
    
    VectorXreal ubu(0);
    
    
    SQP_NLP ocp_solver(0,2, 0,1,1);
    ocp_solver.set_reference(x_ref);
    ocp_solver.set_initial_guess(x);
    ocp_solver.set_h_bounds(lbh, ubh);
    ocp_solver.set_x_bounds(lbx, ubx);
    
    ocp_solver.set_residual_expr(residual);
    ocp_solver.set_g_expr(g_constraints);
    ocp_solver.set_h_expr(h_constraints);

    
    ocp_solver.solve(100,1e-6,1);
    auto solution = ocp_solver.get_solution();

    std::cout << "Solution: " << solution << "\n";

}
