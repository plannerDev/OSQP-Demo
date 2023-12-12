#pragma once

#include <assert.h>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <matplot/matplot.h>

#include "kinematic_trajectory_optimization.h"

namespace gsmpl
{
    void matrixToArray(const Eigen::MatrixXd &m, OSQPFloat *a)
    {
        assert(m.cols() == 1);
        for (int i = 0; i < m.rows(); i++)
            a[i] = m(i, 0);
    }
    std::shared_ptr<OSQPCscMatrix> eigenSparseMatrixToCsc(
        Eigen::SparseMatrix<double> &A)
    {
        assert(A.coeffs().size() == A.nonZeros());
        if (!A.isCompressed())
            A.makeCompressed();
        int rows = A.rows();
        int cols = A.cols();
        int num_data = A.nonZeros();

        std::shared_ptr<OSQPCscMatrix> csc = std::make_shared<OSQPCscMatrix>();
        csc->m = rows;
        csc->n = cols;
        csc->nzmax = num_data;
        csc->nz = 0;
        csc->p = static_cast<OSQPInt *>(malloc((cols + 1) * sizeof(OSQPInt)));
        csc->i = static_cast<OSQPInt *>(malloc(num_data * sizeof(OSQPInt)));
        csc->x = static_cast<OSQPFloat *>(malloc(num_data * sizeof(OSQPFloat)));

        int p_index = 0;
        for (int k = 0; k < cols; k++)
        {
            csc->p[p_index] = static_cast<OSQPInt>(A.outerIndexPtr()[k]);
            p_index++;
        }
        csc->p[p_index] = num_data;

        for (int i = 0; i < num_data; i++)
        {
            csc->i[i] = A.innerIndexPtr()[i];
            csc->x[i] = A.coeffs()[i];
        }
        return csc;
    }

    Eigen::SparseMatrix<double> cscToEigenSparseMatrix(
        const std::shared_ptr<OSQPCscMatrix> &csc)
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
    OSQPInt update_data_mat(OSQPSolver *solver, Eigen::SparseMatrix<double> &Px_new,
                            Eigen::SparseMatrix<double> &Ax_new,
                            std::shared_ptr<OSQPCscMatrix> &P_csc,
                            std::shared_ptr<OSQPCscMatrix> &A_csc)
    {
        P_csc = eigenSparseMatrixToCsc(Px_new);
        A_csc = eigenSparseMatrixToCsc(Ax_new);
        OSQPInt P_new_n = static_cast<OSQPInt>(Px_new.nonZeros());
        OSQPInt A_new_n = static_cast<OSQPInt>(Ax_new.nonZeros());
        return osqp_update_data_mat(solver, P_csc->x, OSQP_NULL, P_new_n, A_csc->x,
                                    OSQP_NULL, A_new_n);
    }
    void printCsc(const std::shared_ptr<OSQPCscMatrix> csc,
                  const std::string &desc)
    {
        std::cout << "CSC Matrix " << desc << std::endl;

        std::cout << "row: " << csc->m << " col: " << csc->n << " nzmax "
                  << csc->nzmax << " triplet " << csc->nz << std::endl;
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
    void printEigenSparseMatrix(const Eigen::SparseMatrix<double> &A,
                                const std::string &desc)
    {
        std::cout << "The matrix " << desc << " is:" << std::endl
                  << Eigen::MatrixXd(A) << std::endl;
        std::cout << "it has " << A.nonZeros()
                  << " stored non zero coefficients that are: "
                  << A.coeffs().transpose() << std::endl;
    }

    BsplineTrajectory KinematicTrajectoryOpti::solve()
    {
        // for test data
        traj_ = traj();
        // end test data
        int num_of_constraints = (4 + 2 * num_control_point_) * dim_;
        int num_of_variables = 1 + num_control_point_ * dim_;
        // Q
        double Qt = 5.0; // TODO
        Eigen::MatrixXd r = Eigen::MatrixXd::Identity(dim_, dim_);
        Eigen::MatrixXd eye =
            Eigen::MatrixXd::Identity(num_control_point_, num_control_point_);
        Eigen::MatrixXd R = kroneckerProduct_eye(3 * eye, r);
        std::cout << "R " << R.rows() << " " << R.cols() << std::endl;
        Eigen::MatrixXd H =
            Eigen::MatrixXd::Identity(num_of_variables, num_of_variables);
        H(0, 0) = Qt;
        H.block(1, 1, R.rows(), R.cols()) = R;
        std::cout << "H " << H.rows() << " " << H.cols() << std::endl;
        // P_reference
        Eigen::VectorXd Pr = Eigen::VectorXd::Zero(num_control_point_ * dim_);
        std::cout << "Pr " << Pr.rows() << " " << Pr.cols() << std::endl;
        for (int i = 0; i < num_control_point_; i++)
            Pr.block(i * dim_, 0, dim_, 1) = control_points_[i];
        std::cout << Pr.transpose() << std::endl;
        // q
        Eigen::MatrixXd q1 = Eigen::MatrixXd::Zero(1, 1);
        Eigen::MatrixXd q2 = -R * Pr;
        Eigen::VectorXd q = vstack(q1, q2).col(0);
        std::cout << "q " << q.size() << std::endl;
        // std::cout << q.transpose() << std::endl;
        // Aq
        Eigen::MatrixXd Aq0 = hstack(Eigen::MatrixXd::Zero(dim_, 1), Bt(0));
        Eigen::MatrixXd Aq1 = hstack(Eigen::MatrixXd::Zero(dim_, 1), Bt(1));
        std::cout << "Aq0 " << Aq0.rows() << " " << Aq0.cols() << std::endl;
        // std::cout << Aq0 << std::endl;
        std::cout << "Aq1 " << Aq1.rows() << " " << Aq1.cols() << std::endl;
        // std::cout << Aq1 << std::endl;
        // M
        Eigen::MatrixXd M1_A = Eigen::MatrixXd::Identity(num_control_point_ - 1,
                                                         num_control_point_ - 1);
        Eigen::MatrixXd M1 =
            kroneckerProduct_eye(M1_A, Eigen::MatrixXd::Identity(dim_, dim_));
        // std::cout << "M1 " << M1.rows() << " " << M1.cols() << std::endl;
        // std::cout << M1 << std::endl;

        Eigen::MatrixXd M2_A = subDiagIdentyMatrix(num_control_point_ - 1);
        Eigen::Matrix M2 =
            kroneckerProduct_subEye(M2_A, Eigen::MatrixXd::Identity(dim_, dim_));
        // std::cout << "M2 " << M2.rows() << " " << M2.cols() << std::endl;
        // std::cout << M2 << std::endl;

        Eigen::MatrixXd M3 = M1 - M2;
        // std::cout << "M3 " << M3.rows() << " " << M3.cols() << std::endl;
        // std::cout << M3 << std::endl;

        Eigen::MatrixXd M4 =
            Eigen::MatrixXd::Zero((num_control_point_ - 1) * dim_, dim_ + 1);
        M4.block(0, 1, dim_, dim_) = -1 * Eigen::MatrixXd::Identity(dim_, dim_);
        Eigen::MatrixXd M = hstack(M4, M3);
        std::cout << "M " << M.rows() << " " << M.cols() << std::endl;
        // std::cout << M << std::endl;

        // Adq0, Adq1
        Eigen::MatrixXd Adq0 = dBcoe(0) * M;
        std::cout << "Adq0 " << Adq0.rows() << " " << Adq0.cols() << std::endl;
        // std::cout << Adq0 << std::endl;
        Eigen::MatrixXd Adq1 = dBcoe(1) * M;
        std::cout << "Adq1 " << Adq1.rows() << " " << Adq1.cols() << std::endl;
        // std::cout << Adq1 << std::endl;

        // Aq = [Aq0; Aq1; Adq0; Adq1]
        Eigen::MatrixXd Aq = vstack(Aq0, Aq1);
        Aq = vstack(Aq, Adq0);
        Aq = vstack(Aq, Adq1);
        std::cout << "Aq " << Aq.rows() << " " << Aq.cols() << std::endl;
        // std::cout << Aq << std::endl;

        // Lq = Uq = [start_p; goal_p; start_vel; goal_vel]
        Eigen::VectorXd start_p = control_points_[0];    // TODO
        Eigen::VectorXd goal_p = control_points_.back(); // TODO
        Eigen::VectorXd start_vel = Eigen::VectorXd::Constant(dim_, 0);
        Eigen::VectorXd goal_vel = Eigen::VectorXd::Constant(dim_, 0);
        // std::cout << "start_p" << start_p.size() << std::endl;
        // std::cout << start_p << std::endl;
        // std::cout << "goal_p" << goal_p.size() << std::endl;
        // std::cout << goal_p << std::endl;
        Eigen::MatrixXd Lq = vstack(start_p.col(0), goal_p.col(0));
        Lq = vstack(Lq, start_vel.col(0));
        Lq = vstack(Lq, goal_vel.col(0));
        std::cout << "Lq " << Lq.rows() << " " << Lq.cols() << std::endl;
        std::cout << Lq.transpose() << std::endl;
        Eigen::MatrixXd Uq = Lq;

        // dM
        Eigen::MatrixXd dM = Eigen::MatrixXd::Zero(1, num_of_variables);
        dM(0, 0) = 1;
        std::cout << "dM " << dM.rows() << " " << dM.cols() << std::endl;
        // std::cout << dM << std::endl;
        // dqmin, dqmax
        Eigen::VectorXd dqmin = Eigen::VectorXd::Constant(dim_, -1);
        Eigen::VectorXd dqmax = Eigen::VectorXd::Constant(dim_, 1);
        // dQmin, dQmax
        Eigen::VectorXd dQmin, dQmax;
        for (int i = 0; i < num_vel_constraint_; i++)
        {
            if (dQmin.cols() == 0)
                dQmin = vstack(dqmin.col(0), dqmin.col(0));
            else
                dQmin = vstack(dQmin.col(0), dqmin.col(0));
        }
        for (int i = 0; i < num_vel_constraint_; i++)
        {
            if (dQmax.cols() == 0)
                dQmax = vstack(dqmax.col(0), dqmax.col(0));
            else
                dQmax = vstack(dQmax.col(0), dqmax.col(0));
        }
        std::cout << "dQmin " << dQmin.rows() << " " << dQmin.cols() << std::endl;
        std::cout << dQmin.transpose() << std::endl;
        std::cout << "dQmax " << dQmax.rows() << " " << dQmax.cols() << std::endl;
        std::cout << dQmax.transpose() << std::endl;

        // dBcoeS = [dBcoe(0), dBcoe(0.25), dBcoe(0.5), dBcoe(0.75), dBcoe(1)]
        // AiqL = dBcoeS * M - dQmin * dM >= 0
        // AiqU = dBcoeS * M - dQmax * dM <= 0
        Eigen::MatrixXd dBcoeS = dBcoe_sequence();
        Eigen::MatrixXd AiqL = dBcoeS * M - dQmin * dM;
        Eigen::MatrixXd AiqU = dBcoeS * M - dQmax * dM;
        std::cout << "AiqL " << AiqL.rows() << " " << AiqL.cols() << std::endl;
        // std::cout << AiqL << std::endl;
        std::cout << "AiqU " << AiqU.rows() << " " << AiqU.cols() << std::endl;
        // std::cout << AiqU << std::endl;

        Eigen::MatrixXd A = vstack(Aq, AiqL);
        A = vstack(A, AiqU);
        std::cout << "A " << A.rows() << " " << A.cols() << std::endl;
        // std::cout << A << std::endl;

        Eigen::VectorXd L = vstack(Lq, Eigen::VectorXd::Zero(AiqL.rows()));
        double inf = std::numeric_limits<double>::infinity();
        L = vstack(L, Eigen::VectorXd::Constant(AiqU.rows(), -inf));
        Eigen::VectorXd U = vstack(Uq, Eigen::VectorXd::Constant(AiqL.rows(), inf));
        U = vstack(U, Eigen::VectorXd::Zero(AiqU.rows()));

        std::cout << "L " << L.rows() << " " << L.cols() << std::endl;
        std::cout << L.transpose() << std::endl;
        std::cout << "U " << U.rows() << " " << U.cols() << std::endl;
        std::cout << U.transpose() << std::endl;

        // OSQP data
        Eigen::SparseMatrix<double> H_sparse = H.sparseView();
        std::shared_ptr<OSQPCscMatrix> H_osqp = eigenSparseMatrixToCsc(H_sparse);
        std::cout << "H_osqp " << H_osqp->nzmax << std::endl;
        OSQPFloat q_osqp[q.rows()];
        matrixToArray(q, q_osqp);
        Eigen::SparseMatrix<double> A_sparse = A.sparseView();
        std::shared_ptr<OSQPCscMatrix> A_osqp = eigenSparseMatrixToCsc(A_sparse);
        OSQPFloat L_osqp[L.rows()];
        matrixToArray(L, L_osqp);
        OSQPFloat U_osqp[U.rows()];
        matrixToArray(U, U_osqp);
        // OSQP solver
        OSQPSolver *solver;
        OSQPSettings settings;
        osqp_set_default_settings(&settings);
        settings.alpha = 1.0;

        OSQPInt exitflag =
            osqp_setup(&solver, H_osqp.get(), q_osqp, A_osqp.get(), L_osqp, U_osqp,
                       A.rows(), num_of_variables, &settings);
        exitflag = osqp_solve(solver);
        std::cout << "solution: x ";
        for (int i = 0; i < num_of_variables; i++)
            std::cout << solver->solution->x[i] << " ";
        std::cout << std::endl;

        // control points
        std::vector<Eigen::VectorXd> control_points;
        for (int i = 0; i < num_control_point_; i++)
        {
            Eigen::VectorXd p = Eigen::VectorXd::Zero(dim_);
            for (int j = 0; j < dim_; j++)
                p(j) = solver->solution->x[i * dim_ + j + 1];
            control_points.push_back(p);
        }

        return BsplineTrajectory(BsplineBasis{traj_.basis().order(), num_control_point_, KnotVectorType::kClamptedUniform, 0, solver->solution->x[0]},
                                 control_points);
    }

    void KinematicTrajectoryOpti::plot_traj()
    {
        traj_ = traj();

        double dt = 1 / num_control_point_;
        std::vector<double> t0 = matplot::linspace(0, 1, num_control_point_);
        std::vector<double> t1 = matplot::linspace(0, 1, 100);
        std::vector<double> q0_v;
        std::vector<double> q1_v;
        std::vector<double> c0_v;
        std::vector<double> c1_v;

        for (int i = 0; i < t0.size(); i++)
        {
            c0_v.push_back(control_points_[i](0));
            c1_v.push_back(control_points_[i](1));
        }

        for (int i = 0; i < t1.size(); i++)
        {
            q0_v.push_back(traj_.value(t1[i])(0));
            q1_v.push_back(traj_.value(t1[i])(1));
        }

        std::cout << "t0 " << t0.size() << " q0 " << q0_v.size() << std::endl;
        std::cout << "t1 " << t1.size() << " q1 " << q1_v.size() << std::endl;
        matplot::plot(c0_v, c1_v, q0_v, q1_v, "--");

        solve();
        matplot::show();
    }

} // namespace gsmpl
