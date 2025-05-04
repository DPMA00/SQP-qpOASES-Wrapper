# SQP qpOASES Wrapper for Nonlinear Programming

This is a lightweight C++ testmodule developed with the objective of deepening my understanding of C++ OOP, and numerical optimal control.

As of now the module supports solving simple nonlinear optimization problems using Sequential Quadratic Programming (SQP) with `qpoases` as the QP solver and `autodiff` as an automatic differentiation tool. The objective function should be able to be written in least squares form such that a Gauss-Newton Hessian approximation can be used.

Planned features include support for multiple shooting discretization for optimal control (and optional condensing), and possibly a globalization strategy.