#include "SQP_NLP.h"



void testcase3()
{
    auto h_constraints = [](const VectorXreal& z) -> VectorXreal 
    {
        VectorXreal r(2);
        r(0) = z(0) + z(1) + z(2) + z(3);
        r(1) = pow(z(0),2) + pow(z(1),2) + pow(z(2),2) + pow(z(3),2);
        return r;
    };
    

    auto g_constraints= [] (const VectorXreal &z) -> VectorXreal
    {
        VectorXreal r(1);
        r(0) = z(0)*z(3) - z(1)*z(2);
        return r;
    };

    auto residual = [](const VectorXreal &z, const VectorXreal &z_ref) -> VectorXreal
    {
        VectorXreal r(3);
        r(0) = sqrt(2)*(z(0)-1);
        r(1) = sqrt(2)*(z(1)-z(2));
        r(2) = sqrt(2)*(z(3)-1);
        
        return r;
    };


    VectorXreal x(4);
    x << 0,0,0,0;

    VectorXreal x_ref(4);
    x_ref << 0,0,0,0;

    VectorXreal lbh(2);
    lbh << -1e5, -1e5;

    VectorXreal ubh(2);
    ubh << 8, 25;

    VectorXreal lbx(4);
    lbx << -1e5, -1e5, -1e5, -1e5;
    
    VectorXreal ubx(4);
    ubx << 1e5, 1e5, 1e5, 1e5;

    VectorXreal lbu(0);
    
    VectorXreal ubu(0);
    
    
    SQP_NLP ocp_solver(0,4, 0,1,2);
    ocp_solver.debugger_ = false;
    ocp_solver.set_reference(x_ref);
    ocp_solver.set_initial_guess(x);
    ocp_solver.set_h_bounds(lbh, ubh);
    ocp_solver.set_x_bounds(lbx, ubx);
    
    ocp_solver.set_residual_expr(residual);
    ocp_solver.set_g_expr(g_constraints);
    ocp_solver.set_h_expr(h_constraints);

   
    ocp_solver.solve(100,1e-6,1);
    auto solution = ocp_solver.get_solution();

    std::cout << "Solution testcase 3: " << "\n" << solution << "\n";
 
}

int main()
{
    testcase3();
}