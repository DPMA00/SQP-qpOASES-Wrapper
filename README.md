# SQP qpOASES Wrapper for Nonlinear Programming

This is a lightweight C++ testmodule developed with the objective of deepening my understanding of C++ OOP, and numerical optimal control.

As of now the module supports solving simple nonlinear optimization problems using Sequential Quadratic Programming (SQP) with `qpoases` as the QP solver and `autodiff` as an automatic differentiation tool. 

The objective function should be able to be written in least squares form such that a Gauss-Newton Hessian approximation can be used.

```math 
f(w_k) = \| R(w_k) \|_2² 
```
```math
B_k = J(R(w_k))J(R(w_k))^T
```

Forming the QP subproblem:

```math
\begin{align*}
\underset{\Delta w}{\text{min}} \qquad &\frac{1}{2}\Delta w^T B_k \Delta w + \nabla f(w_k)^T \Delta w\\
\text{s.t. } \qquad&-g(w_k)\leq J(g(w_k)) \Delta w \leq -g(w_k),\\
&-h(w_k)+lb_h\leq J(h(w_k)) \Delta w \leq  -h(w_k) + ub_h,\\
&lb_w \leq \Delta w \leq ub_w,
\end{align*}
```
Which respects the standard form qpOASES expects.

Planned features include support for multiple shooting discretization for optimal control (and optional condensing), and possibly a globalization strategy.