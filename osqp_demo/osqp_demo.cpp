#include <iostream>
#include <math.h>
#include "osqp_demo.h"
#include <osqp/osqp.h>

void matrixToArray(const Eigen::MatrixXd &m, OSQPFloat *a)
{
  assert(m.cols() == 1);
  for (int i = 0; i < m.rows(); i++)
    a[i] = m(i, 0);
}
std::shared_ptr<OSQPCscMatrix> eiegnSparseMatrixToCsc(Eigen::SparseMatrix<double> &sparse_matrix)
{
  assert(sparse_matrix.coeffs().size() == sparse_matrix.nonZeros());
  if (!sparse_matrix.isCompressed())
    sparse_matrix.makeCompressed();
  int rows = sparse_matrix.rows();
  int cols = sparse_matrix.cols();
  int num_data = sparse_matrix.nonZeros();

  OSQPCscMatrix *csc = csc_spalloc(static_cast<OSQPInt>(rows),
                                   static_cast<OSQPInt>(cols), static_cast<OSQPInt>(num_data), 1, 0);
  int p_index = 0;
  for (int k = 0; k < cols; k++)
  {
    csc->p[p_index] = static_cast<OSQPInt>(sparse_matrix.outerIndexPtr()[p_index]);
    p_index++;
  }
  csc->p[p_index] = num_data;

  for (int i = 0; i < num_data; i++)
  {
    csc->i[i] = sparse_matrix.innerIndexPtr()[i];
    csc->x[i] = sparse_matrix.coeffs()[i];
  }
  return std::make_shared<OSQPCscMatrix>(*csc);
}

Eigen::SparseMatrix<double> cscToEiegnSparseMatrix(const std::shared_ptr<OSQPCscMatrix> &csc)
{
  int rows = csc->m;
  int cols = csc->n;
  Eigen::SparseMatrix<double> A(rows, cols);

  int data_index = 0;
  for (int col = 0; col < cols; col++)
  {
    int num = csc->p[col + 1] - csc->p[col];
    for (int i = 0; i < num; i++)
    {
      A.insert(csc->i[data_index], col) = csc->x[data_index];
      data_index++;
    }
  }

  // for (int i = 0; i < csc->nzmax - data_index; i++)
  // {
  //   A.insert(csc->i[data_index], cols - 1) = csc->x[data_index];
  //   data_index++;
  // }
  assert(data_index == csc->nzmax);
  A.makeCompressed();
  return A;
}
OSQPInt update_data_mat(OSQPSolver *solver,
                        Eigen::SparseMatrix<double> &Px_new,
                        Eigen::SparseMatrix<double> &Ax_new,
                        std::shared_ptr<OSQPCscMatrix> &P_csc,
                        std::shared_ptr<OSQPCscMatrix> &A_csc)
{
  P_csc = eiegnSparseMatrixToCsc(Px_new);
  A_csc = eiegnSparseMatrixToCsc(Ax_new);
  OSQPInt P_new_n = static_cast<OSQPInt>(Px_new.nonZeros());
  OSQPInt A_new_n = static_cast<OSQPInt>(Ax_new.nonZeros());
  osqp_update_data_mat(solver, P_csc->x, OSQP_NULL, P_new_n, A_csc->x, OSQP_NULL, A_new_n);
}
void printCsc(const std::shared_ptr<OSQPCscMatrix> csc,
              const std::string &desc)
{
  std::cout << "CSC Matrix " << desc << std::endl;

  std::cout << "row: " << csc->m << " col: " << csc->n
            << " nzmax " << csc->nzmax << " triplet " << csc->nz << std::endl;
  std::cout << "x: " << std::endl;
  for (int i = 0; i < csc->nzmax; i++)
    std::cout << csc->x[i] << " ";
  std::cout << std::endl;

  std::cout << "col: " << std::endl;
  for (int i = 0; i < csc->n + 1; i++)
    std::cout << csc->p[i] << " ";
  std::cout << std::endl;

  std::cout << "row: " << std::endl;
  for (int i = 0; i < csc->nzmax; i++)
    std::cout << csc->i[i] << " ";
  std::cout << std::endl;
}
void printEigenSparseMatrix(const Eigen::SparseMatrix<double> &A, const std::string &desc)
{
  std::cout << "The matrix " << desc << " is:" << std::endl
            << Eigen::MatrixXd(A) << std::endl;
  std::cout << "it has " << A.nonZeros() << " stored non zero coefficients that are: "
            << A.coeffs().transpose() << std::endl;
}
void test()
{
  Eigen::SparseMatrix<double> A(3, 4);
  A.insert(0, 0) = 10;
  A.insert(2, 0) = 11;
  A.insert(2, 2) = 8;
  A.insert(0, 3) = 3;
  A.makeCompressed();

  printEigenSparseMatrix(A, "A");
  std::shared_ptr<OSQPCscMatrix> csc = eiegnSparseMatrixToCsc(A);
  printCsc(csc, "A");
  Eigen::SparseMatrix<double> B = cscToEiegnSparseMatrix(csc);
  printEigenSparseMatrix(B, "B");
}

int TestCase::setupAndSolve()
{

  Eigen::SparseMatrix<double> P(2, 2);
  P.insert(0, 0) = 4;
  P.insert(0, 1) = 1;
  // P.insert(1, 0) = 0;
  P.insert(1, 1) = 2;
  P.makeCompressed();
  printEigenSparseMatrix(P, "P");

  Eigen::SparseMatrix<double> A(3, 2);
  A.insert(0, 0) = 1;
  A.insert(1, 0) = 1;
  A.insert(0, 1) = 1;
  A.insert(2, 1) = 1;
  A.makeCompressed();
  printEigenSparseMatrix(A, "A");

  std::shared_ptr<OSQPCscMatrix> P_csc = eiegnSparseMatrixToCsc(P);
  printCsc(P_csc, "P");
  std::shared_ptr<OSQPCscMatrix> A_csc = eiegnSparseMatrixToCsc(A);
  printCsc(A_csc, "A");

  OSQPFloat q[2] = {1, 1};
  OSQPFloat l[3] = {1, 0, 0};
  OSQPFloat u[3] = {1, 0.7, 0.7};

  OSQPInt m = 3; // num of constraint
  OSQPInt n = 2; // num of variable

  OSQPSolver *solver;
  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.alpha = 1.0;

  OSQPInt exitflag = osqp_setup(&solver, P_csc.get(), q, A_csc.get(), l, u,
                                m, n, &settings);
  if (!exitflag)
  {
    exitflag = osqp_solve(solver);
    if (!exitflag)
    {
      std::cout << "Variable: ";
      for (int i = 0; i < n; i++)
        std::cout << solver->solution->x[i] << " ";
      std::cout << std::endl;
    }
  }

  osqp_cleanup(solver);
  if (P_csc.get())
    csc_spfree(P_csc.get());
  if (A_csc.get())
    csc_spfree(A_csc.get());

  return static_cast<int>(exitflag);
}
int TestCase::updateVectors()
{
  Eigen::SparseMatrix<double> P(2, 2);
  P.insert(0, 0) = 4;
  P.insert(0, 1) = 1;
  // P.insert(1, 0) = 0;
  P.insert(1, 1) = 2;
  P.makeCompressed();
  printEigenSparseMatrix(P, "P");

  Eigen::SparseMatrix<double> A(3, 2);
  A.insert(0, 0) = 1;
  A.insert(1, 0) = 1;
  A.insert(0, 1) = 1;
  A.insert(2, 1) = 1;
  A.makeCompressed();
  printEigenSparseMatrix(A, "A");

  std::shared_ptr<OSQPCscMatrix> P_csc = eiegnSparseMatrixToCsc(P);
  printCsc(P_csc, "P");
  std::shared_ptr<OSQPCscMatrix> A_csc = eiegnSparseMatrixToCsc(A);
  printCsc(A_csc, "A");

  OSQPFloat q[2] = {1, 1};
  OSQPFloat l[3] = {1, 0, 0};
  OSQPFloat u[3] = {1, 0.7, 0.7};

  OSQPFloat q_new[2] = {2, 3};
  OSQPFloat l_new[3] = {2, -1, -1};
  OSQPFloat u_new[3] = {2, 2.5, 2.5};

  OSQPInt m = 3; // num of constraint
  OSQPInt n = 2; // num of variable

  OSQPSolver *solver;
  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.alpha = 1.0;

  OSQPInt exitflag = osqp_setup(&solver, P_csc.get(), q, A_csc.get(), l, u,
                                m, n, &settings);
  if (!exitflag)
  {
    exitflag = osqp_solve(solver);
    if (!exitflag)
    {
      std::cout << "Variable: ";
      for (int i = 0; i < n; i++)
        std::cout << solver->solution->x[i] << " ";
      std::cout << std::endl;
    }
  }

  exitflag = osqp_update_data_vec(solver, q_new, l_new, u_new);
  if (!exitflag)
  {
    exitflag = osqp_solve(solver);
    if (!exitflag)
    {
      std::cout << "updated vec Variable: ";
      for (int i = 0; i < n; i++)
        std::cout << solver->solution->x[i] << " ";
      std::cout << std::endl;
    }
  }

  osqp_cleanup(solver);
  if (P_csc.get())
    csc_spfree(P_csc.get());
  if (A_csc.get())
    csc_spfree(A_csc.get());

  return static_cast<int>(exitflag);
}
int TestCase::updateMatrix()
{
  Eigen::SparseMatrix<double> P(2, 2);
  P.insert(0, 0) = 4;
  P.insert(0, 1) = 1;
  // P.insert(1, 0) = 0;
  P.insert(1, 1) = 2;
  P.makeCompressed();
  printEigenSparseMatrix(P, "P");

  Eigen::SparseMatrix<double> A(3, 2);
  A.insert(0, 0) = 1;
  A.insert(1, 0) = 1;
  A.insert(0, 1) = 1;
  A.insert(2, 1) = 1;
  A.makeCompressed();
  printEigenSparseMatrix(A, "A");

  std::shared_ptr<OSQPCscMatrix> P_csc = eiegnSparseMatrixToCsc(P);
  printCsc(P_csc, "P");
  std::shared_ptr<OSQPCscMatrix> A_csc = eiegnSparseMatrixToCsc(A);
  printCsc(A_csc, "A");

  OSQPFloat q[2] = {1, 1};
  OSQPFloat l[3] = {1, 0, 0};
  OSQPFloat u[3] = {1, 0.7, 0.7};

  OSQPInt m = 3; // num of constraint
  OSQPInt n = 2; // num of variable

  OSQPSolver *solver;
  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.alpha = 1.0;

  OSQPInt exitflag = osqp_setup(&solver, P_csc.get(), q, A_csc.get(), l, u,
                                m, n, &settings);
  if (!exitflag)
  {
    exitflag = osqp_solve(solver);
    if (!exitflag)
    {
      std::cout << "Variable: ";
      for (int i = 0; i < n; i++)
        std::cout << solver->solution->x[i] << " ";
      std::cout << std::endl;
    }
  }
  Eigen::SparseMatrix<double> P_new(2, 2);
  P_new.insert(0, 0) = 5;
  P_new.insert(0, 1) = 1.5;
  P_new.insert(1, 1) = 1;
  P_new.makeCompressed();
  printEigenSparseMatrix(P_new, "P_new");

  Eigen::SparseMatrix<double> A_new(3, 2);
  A_new.insert(0, 0) = 1.2;
  A_new.insert(1, 0) = 1.5;
  A_new.insert(0, 1) = 1.1;
  A_new.insert(2, 1) = 0.8;
  A_new.makeCompressed();
  printEigenSparseMatrix(A_new, "A_new");

  std::shared_ptr<OSQPCscMatrix> P_new__csc(OSQP_NULL);
  std::shared_ptr<OSQPCscMatrix> A_new_csc(OSQP_NULL);
  update_data_mat(solver, P_new, A_new, P_new__csc, A_new_csc);

  if (!exitflag)
  {
    exitflag = osqp_solve(solver);
    if (!exitflag)
    {
      std::cout << "updated mat Variable: ";
      for (int i = 0; i < n; i++)
        std::cout << solver->solution->x[i] << " ";
      std::cout << std::endl;
    }
  }

  osqp_cleanup(solver);
  if (P_csc.get())
    csc_spfree(P_csc.get());
  if (A_csc.get())
    csc_spfree(A_csc.get());

  return static_cast<int>(exitflag);
}
Eigen::MatrixXd kroneckerProduct_eye(const Eigen::MatrixXd &lhs,
                                     const Eigen::MatrixXd &rhs)
{
  Eigen::MatrixXd out(lhs.rows() * rhs.rows(), lhs.cols() * rhs.cols());
  out.setZero();
  for (int i = 0; i < lhs.rows(); i++)
  {
    out.block(i * rhs.rows(), i * rhs.cols(), rhs.rows(), rhs.cols()) = lhs(i, i) * rhs;
  }
  // std::cout << "kroneckerProduct_eye size " << out.rows() << " " << out.cols() << "\n"
  //           << out << std::endl;
  return out;
}
Eigen::MatrixXd kroneckerProduct_subEye(const Eigen::MatrixXd &lhs,
                                        const Eigen::MatrixXd &rhs)
{
  Eigen::MatrixXd out(lhs.rows() * rhs.rows(), lhs.cols() * rhs.cols());
  out.setZero();
  for (int i = 1; i < lhs.rows(); i++)
  {
    out.block(i * rhs.rows(), (i - 1) * rhs.cols(), rhs.rows(), rhs.cols()) = lhs(i, i - 1) * rhs;
  }
  // std::cout << "kroneckerProduct_subEye size " << out.rows() << " " << out.cols() << "\n"
  //           << out << std::endl;
  return out;
}
Eigen::SparseMatrix<double> subDiagIdentyMatrix(int n)
{
  Eigen::SparseMatrix<double> out(n, n);
  for (int i = 1; i < n; i++)
    out.insert(i, i - 1) = 1.0;

  out.finalize();

  // std::cout << "subDiagIdentyMatrix \n"
  //           << out.toDense() << std::endl;
  return out;
}
// [A B]
Eigen::MatrixXd hstack(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
{
  assert(A.rows() == B.rows());
  Eigen::MatrixXd out(A.rows(), A.cols() + B.cols());
  out.setZero();
  out.block(0, 0, A.rows(), A.cols()) = A;
  out.block(0, A.cols(), B.rows(), B.cols()) = B;
  // std::cout << "hstack size " << out.rows() << " " << out.cols() << "\n"
  //           << out << std::endl;
  return out;
}

// [A; B]
Eigen::MatrixXd vstack(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
{
  assert(A.cols() == B.cols());
  Eigen::MatrixXd out(A.rows() + B.rows(), A.cols());
  out.setZero();
  out.block(0, 0, A.rows(), A.cols()) = A;
  out.block(A.rows(), 0, B.rows(), B.cols()) = B;
  // std::cout << "vstack size " << out.rows() << " " << out.cols() << "\n"
  //           << out << std::endl;
  return out;
}
Eigen::SparseMatrix<double> MPC::create_Ad()
{
  Eigen::SparseMatrix<double> Ad(12, 12);
  Ad.insert(0, 0) = 1;
  Ad.insert(0, 6) = 0.1;
  Ad.insert(1, 1) = 1;
  Ad.insert(1, 7) = 0.1;
  Ad.insert(2, 2) = 1;
  Ad.insert(2, 8) = 0.1;
  Ad.insert(3, 0) = 0.0488;
  Ad.insert(3, 3) = 1;
  Ad.insert(3, 6) = 0.0016;
  Ad.insert(3, 9) = 0.0992;
  Ad.insert(4, 1) = -0.0488;
  Ad.insert(4, 4) = 1;
  Ad.insert(4, 7) = -0.0016;
  Ad.insert(4, 10) = 0.0992;
  Ad.insert(5, 5) = 1;
  Ad.insert(5, 11) = 0.0092;
  Ad.insert(6, 6) = 1;
  Ad.insert(7, 7) = 1;
  Ad.insert(8, 8) = 1;
  Ad.insert(9, 0) = 0.9734;
  Ad.insert(9, 6) = 0.0488;
  Ad.insert(9, 9) = 0.9846;
  Ad.insert(10, 1) = -0.9734;
  Ad.insert(10, 7) = -0.0488;
  Ad.insert(10, 10) = 0.9846;
  Ad.insert(11, 11) = 0.9846;
  Ad.makeCompressed();
  printEigenSparseMatrix(Ad, "Ad");
  return Ad;
}

Eigen::SparseMatrix<double> MPC::create_Bd()
{
  Eigen::SparseMatrix<double> Bd(12, 4);
  Bd.insert(0, 1) = -0.0726;
  Bd.insert(0, 3) = 0.0726;
  Bd.insert(1, 0) = -0.0726;
  Bd.insert(1, 2) = 0.0726;
  Bd.insert(2, 0) = -0.0152;
  Bd.insert(2, 1) = 0.0152;
  Bd.insert(2, 2) = -0.0152;
  Bd.insert(2, 3) = 0.0152;
  Bd.insert(3, 1) = -0.0006;
  Bd.insert(3, 3) = 0.0006;
  Bd.insert(4, 0) = 0.0006;
  Bd.insert(4, 2) = -0.0006;
  Bd.insert(5, 0) = 0.0106;
  Bd.insert(5, 1) = 0.0106;
  Bd.insert(5, 2) = 0.0106;
  Bd.insert(5, 3) = 0.0106;
  Bd.insert(6, 1) = -1.4512;
  Bd.insert(6, 3) = 1.4512;
  Bd.insert(7, 0) = -1.4512;
  Bd.insert(7, 2) = 1.4512;
  Bd.insert(8, 0) = -0.3049;
  Bd.insert(8, 1) = 0.3049;
  Bd.insert(8, 2) = -0.3049;
  Bd.insert(8, 3) = 0.3049;
  Bd.insert(9, 1) = -0.0236;
  Bd.insert(9, 3) = 0.0236;
  Bd.insert(10, 0) = 0.0236;
  Bd.insert(10, 2) = -0.0236;
  Bd.insert(11, 0) = 0.2107;
  Bd.insert(11, 1) = 0.2107;
  Bd.insert(11, 2) = 0.2107;
  Bd.insert(11, 3) = 0.2107;
  Bd.makeCompressed();
  printEigenSparseMatrix(Bd, "Bd");
  return Bd;
}
Eigen::MatrixXd MPC::create_Q()
{
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> Q(nx_);
  Q.diagonal() << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
  return Q.toDenseMatrix();
}
Eigen::MatrixXd MPC::create_QN()
{
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> QN(nx_);
  QN.diagonal() << 10, 10, 1000, 10, 10, 10, 10, 10, 100, 10, 10, 10;
  return QN.toDenseMatrix();
}
Eigen::MatrixXd MPC::create_R()
{
  return Eigen::MatrixXd::Identity(nu_, nu_) * 0.1;
}

Eigen::MatrixXd MPC::create_P()
{
  Eigen::SparseMatrix<double> speye(N_, N_);
  speye.setIdentity();
  Eigen::MatrixXd P(nVariable_, nVariable_);
  P.setZero();
  P.block(0, 0, N_ * nx_, N_ * nx_) = kroneckerProduct_eye(speye, Q_);
  P.block(N_ * nx_, N_ * nx_, nx_, nx_) = QN_;
  P.block((N_ + 1) * nx_, (N_ + 1) * nx_, N_ * nu_, N_ * nu_) = kroneckerProduct_eye(speye, R_);
  std::cout << "P \n"
            << P << std::endl;
  std::cout << "P size " << P.rows() << " " << P.cols() << std::endl;
  return P;
}
Eigen::MatrixXd MPC::create_q()
{
  Eigen::MatrixXd q(nVariable_, 1);
  q.setZero();
  for (int i = 0; i < N_; i++)
    q.block(i * nx_, 0, nx_, 1) = -2 * Q_ * xr_;
  q.block(N_ * nx_, 0, nx_, 1) = -2 * QN_ * xr_;
  q.block((N_ + 1) * nx_, 0, N_ * nu_, 1) = Eigen::VectorXd::Zero(N_ * nu_);
  std::cout << "q \n"
            << q << std::endl;
  std::cout << "q size " << q.rows() << " " << q.cols() << std::endl;
  return q;
}
Eigen::MatrixXd MPC::create_Ax()
{
  Eigen::SparseMatrix<double> speye_N1(N_ + 1, N_ + 1);
  speye_N1.setIdentity();
  Eigen::SparseMatrix<double> speye_nx(nx_, nx_);
  speye_nx.setIdentity();
  Eigen::SparseMatrix<double> subDiag = subDiagIdentyMatrix(N_ + 1);
  Eigen::MatrixXd Ax = kroneckerProduct_eye(speye_N1, -speye_nx) + kroneckerProduct_subEye(subDiag, Ad_);
  std::cout << "Ax size " << Ax.rows() << " " << Ax.cols() << "\n"
            << Ax << std::endl;
  return Ax;
}
Eigen::MatrixXd MPC::create_Bu()
{
  Eigen::SparseMatrix<double> speye(N_, N_);
  speye.setIdentity();
  Eigen::MatrixXd Bu(nx_ * (N_ + 1), nu_ * N_);
  Bu.setZero();
  // Bu.block(0, nu * (N - 1), nx, nu) = Bd;
  Bu.block(nx_, 0, N_ * nx_, N_ * nu_) = kroneckerProduct_eye(speye, Bd_);
  std::cout << "Bu size " << Bu.rows() << " " << Bu.cols() << "\n"
            << Bu << std::endl;
  return Bu;
}
Eigen::MatrixXd MPC::create_Aeq(const Eigen::MatrixXd &Ax, const Eigen::MatrixXd &Bu)
{
  return hstack(Ax, Bu);
}
Eigen::MatrixXd MPC::create_leq()
{
  Eigen::MatrixXd leq(nx_ * (N_ + 1), 1);
  leq.setZero();
  leq.block(0, 0, nx_, 1) = Eigen::MatrixXd(-x0_);
  std::cout << "leq size " << leq.rows() << " " << leq.cols() << "\n"
            << leq << std::endl;
  return leq;
}
Eigen::MatrixXd MPC::create_Aineq()
{
  Eigen::SparseMatrix<double> Aineq(nVariable_, nVariable_);
  Aineq.setIdentity();
  std::cout << "Aineq size " << Aineq.rows() << " " << Aineq.cols() << "\n"
            << Aineq << std::endl;
  return Aineq;
}
Eigen::MatrixXd MPC::create_lineq()
{
  Eigen::MatrixXd lineq(nVariable_, 1);

  for (int i = 0; i < N_ + 1; i++)
    lineq.block(i * nx_, 0, nx_, 1) = Eigen::MatrixXd(xmin_);
  for (int i = 0; i < N_; i++)
    lineq.block((N_ + 1) * nx_ + i * nu_, 0, nu_, 1) = Eigen::MatrixXd(umin_);
  return lineq;
}
Eigen::MatrixXd MPC::create_uineq()
{
  Eigen::MatrixXd uineq(nVariable_, 1);
  for (int i = 0; i < N_ + 1; i++)
    uineq.block(i * nx_, 0, nx_, 1) = Eigen::MatrixXd(xmax_);
  for (int i = 0; i < N_; i++)
    uineq.block((N_ + 1) * nx_ + i * nu_, 0, nu_, 1) = Eigen::MatrixXd(umax_);
  return uineq;
}
Eigen::VectorXd MPC::create_xmin()
{

  Eigen::VectorXd xmin(nx_);
  xmin << -M_PI / 6, -M_PI / 6, -inf, -inf, -inf, -1,
      -inf, -inf, -inf, -inf, -inf, -inf;
  std::cout << "xmin \n"
            << xmin << std::endl;
  return xmin;
}
Eigen::VectorXd MPC::create_xmax()
{

  Eigen::VectorXd xmax(nx_);
  xmax << M_PI / 6, M_PI / 6, inf, inf, inf, inf,
      inf, inf, inf, inf, inf, inf;
  std::cout << "xmax \n"
            << xmax << std::endl;
  return xmax;
}
Eigen::VectorXd MPC::create_umin()
{
  OSQPFloat u0 = 10.5916;
  Eigen::VectorXd umin(nu_);
  umin << 9.6 - u0, 9.6 - u0, 9.6 - u0, 9.6 - u0;
  std::cout << "umin \n"
            << umin << std::endl;
  return umin;
}
Eigen::VectorXd MPC::create_umax()
{
  OSQPFloat u0 = 10.5916;
  Eigen::VectorXd umax(nu_);
  umax << 13 - u0, 13 - u0, 13 - u0, 13 - u0;
  std::cout << "umax \n"
            << umax << std::endl;
  return umax;
}
Eigen::VectorXd MPC::create_x0()
{
  Eigen::VectorXd x0(nx_);
  x0 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  std::cout << "x0 \n"
            << x0 << std::endl;
  return x0;
}
Eigen::VectorXd MPC::create_xr()
{
  Eigen::VectorXd xr(nx_);
  xr << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  std::cout << "xr \n"
            << xr << std::endl;
  return xr;
}
Eigen::VectorXd create_xr();
MPC::MPC() : N_(10), nsim_(15), nVariable_(N_ * nx_ + nx_ + N_ * nu_)
{

  // Ad Bd
  Ad_ = create_Ad();
  Bd_ = create_Bd();

  // xmin xmax umin umax
  xmin_ = create_xmin();
  xmax_ = create_xmax();
  umin_ = create_umin();
  umax_ = create_umax();

  // x0 xr
  x0_ = create_x0();
  xr_ = create_xr();

  Q_ = create_Q();
  QN_ = create_QN();
  R_ = create_R();
  std::cout << "Q \n"
            << Q_ << std::endl;
  std::cout << "QN \n"
            << QN_ << std::endl;
  std::cout << "R \n"
            << R_ << std::endl;
}
int MPC::solve()
{
  Eigen::MatrixXd P = create_P();
  Eigen::MatrixXd q = create_q();
  Eigen::MatrixXd Ax = create_Ax();
  Eigen::MatrixXd Bu = create_Bu();
  Eigen::MatrixXd Aeq = create_Aeq(Ax, Bu);
  Eigen::MatrixXd leq = create_leq();
  Eigen::MatrixXd ueq = leq;
  Eigen::MatrixXd Aineq = create_Aineq();
  Eigen::MatrixXd lineq = create_lineq();
  Eigen::MatrixXd uineq = create_uineq();

  Eigen::MatrixXd A = vstack(Aeq, Aineq);
  Eigen::MatrixXd l = vstack(leq, lineq);
  Eigen::MatrixXd u = vstack(ueq, uineq);
  // std::cout << "A " << A.rows() << " " << A.cols() << "\n"
  //           << A << "\n"
  //           << "l " << l.rows() << " " << l.cols() << "\n"
  //           << l << "\n"
  //           << "u " << u.rows() << " " << u.cols() << "\n"
  //           << u << std::endl;

  Eigen::SparseMatrix<double> P_sparse = P.sparseView();
  std::shared_ptr<OSQPCscMatrix> P_osqp = eiegnSparseMatrixToCsc(P_sparse);
  std::cout << "P_osqp " << P_osqp->nzmax << std::endl;
  OSQPFloat q_osqp[q.rows()];
  matrixToArray(q, q_osqp);
  Eigen::SparseMatrix<double> A_sparse = A.sparseView();
  std::shared_ptr<OSQPCscMatrix> A_osqp = eiegnSparseMatrixToCsc(A_sparse);
  OSQPFloat l_osqp[l.rows()];
  matrixToArray(l, l_osqp);
  OSQPFloat u_osqp[u.rows()];
  matrixToArray(u, u_osqp);

  // Create an OSQP object
  OSQPSolver *solver;
  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.alpha = 1.0;

  OSQPInt exitflag = osqp_setup(&solver, P_osqp.get(), q_osqp, A_osqp.get(), l_osqp, u_osqp,
                                A.rows(), nVariable_, &settings);
  // std::cout << "************ print **************" << std::endl;
  // std::cout << "P_osqp ";
  // for (int i = 0; i < P_osqp->nzmax; i++)
  //   std::cout << P_osqp->x[i] << " ";
  // std::cout << std::endl;
  // std::cout << "q_osqp ";
  // for (int i = 0; i < q.rows(); i++)
  //   std::cout << q_osqp[i] << " ";
  // std::cout << std::endl;
  // std::cout << "A_osqp ";
  // for (int i = 0; i < A_osqp->nzmax; i++)
  //   std::cout << A_osqp->x[i] << " ";
  // std::cout << std::endl;
  // std::cout << "l_osqp ";
  // for (int i = 0; i < l.rows(); i++)
  //   std::cout << l_osqp[i] << " ";
  // std::cout << std::endl;
  // std::cout << "u_osqp ";
  // for (int i = 0; i < u.rows(); i++)
  //   std::cout << u_osqp[i] << " ";
  // std::cout << std::endl;

  int nsim = 15;
  int u0_index = (N_ + 1) * nx_;
  for (int i = 0; i < nsim; i++)
  {
    exitflag = osqp_solve(solver);
    std::cout << "Variable: ";
    for (int i = 0; i < nx_; i++)
      std::cout << solver->solution->x[i + N_ * nx_] << " ";
    std::cout << std::endl;

    OSQPFloat x0[nx_];
    if (!exitflag)
    {
      OSQPFloat cmd[nu_];
      std::copy(solver->solution->x + u0_index, solver->solution->x + u0_index + nu_, cmd);
      Eigen::Map<Eigen::VectorXd> cmd_vec(cmd, sizeof(cmd) / sizeof(cmd[0]));
      std::cout << "cmd_vec " << cmd_vec << std::endl;
      x0_ = Ad_ * x0_ + Bd_ * cmd_vec;
      std::cout << "updated x0 " << i << " " << x0_ << std::endl;
      matrixToArray(-x0_, x0);
      std::copy(x0, x0 + nx_, u_osqp);
      std::copy(x0, x0 + nx_, l_osqp);
      // std::cout << "l_osqp ";
      // for (int i = 0; i < l.rows(); i++)
      //   std::cout << l_osqp[i] << " ";
      // std::cout << std::endl;
      // std::cout << "u_osqp ";
      // for (int i = 0; i < u.rows(); i++)
      //   std::cout << u_osqp[i] << " ";
      std::cout << std::endl;
      exitflag = osqp_update_data_vec(solver, q_osqp, l_osqp, u_osqp);
    }
  }
}
