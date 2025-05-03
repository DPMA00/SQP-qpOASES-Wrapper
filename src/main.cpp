#include "SQP_NLP.h"

int main()
{
    
    VectorXreal x(2);
    x << 1,-1;

    VectorXreal x_ref(2);
    x_ref << 0,0;


    SQP_NLP NLP_solver(x, x_ref);
    NLP_solver.solve(100,1e-6);

}