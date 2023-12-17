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

    // test data
    std::vector<Eigen::VectorXd> control_points()
    {
        std::vector<Eigen::VectorXd> out;
        Eigen::MatrixXd out_mat(10, 7);
        out_mat << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
            0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
            0.3, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
            0.4, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38,
            0.5, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38,
            0.6, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
            0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
            0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
            0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;

        for (int i = 0; i < out_mat.rows(); i++)
            out.push_back(out_mat.row(i));

        return out;
    }
    Eigen::MatrixXd KinematicTrajectoryOpti::dB_weighted_sequence() const
    {
        Eigen::MatrixXd out;

        double dt = traj_.endTime() / static_cast<double>(n_inequ_constraint_); // TODO: dt

        Eigen::MatrixXd dBcoe_0 = traj_.basis().dBt_weighted(dim_, 0);
        Eigen::MatrixXd dBcoe_1 = traj_.basis().dBt_weighted(dim_, dt);
        out = vstack(dBcoe_0, dBcoe_1);
        for (int i = 2; i < n_inequ_constraint_; i++)
            out = vstack(out, traj_.basis().dBt_weighted(dim_, dt * i));
        std::cout << "dB_weighted_sequence " << out.rows() << " " << out.cols() << std::endl;
        return out;
    }
    void KinematicTrajectoryOpti::init(int n_equa_constraint, int n_inequ_constraint,
                                       const std::vector<Eigen::VectorXd> &contorl_points)
    {
        dim_ = contorl_points[0].size();
        n_ = contorl_points.size();
        nx_ = n_ * dim_ + 1;
        n_equa_constraint_ = n_equa_constraint,
        n_inequ_constraint_ = n_inequ_constraint;
        nc_ = n_equa_constraint_ + n_inequ_constraint_ * 2;
        control_points_ = contorl_points;

        double inf = std::numeric_limits<double>::infinity();

        p_start_ = control_points_[0];
        p_goal_ = control_points_.back();
        vel_start_ = Eigen::VectorXd::Constant(dim_, 0);
        vel_goal_ = Eigen::VectorXd::Constant(dim_, 0);

        vel_min_ = Eigen::VectorXd::Constant(dim_, -0.1);
        vel_max_ = Eigen::VectorXd::Constant(dim_, 0.1);
        acc_min_ = Eigen::VectorXd::Constant(dim_, -inf);
        acc_max_ = Eigen::VectorXd::Constant(dim_, inf);
    }
    Eigen::MatrixXd KinematicTrajectoryOpti::create_R(double w) const
    {
        Eigen::MatrixXd r = Eigen::MatrixXd::Identity(dim_, dim_);
        Eigen::MatrixXd eye =
            Eigen::MatrixXd::Identity(n_, n_);
        Eigen::MatrixXd R = kroneckerProduct_eye(w * eye, r); // weight_r = 3
        std::cout << "R " << R.rows() << " " << R.cols() << std::endl;
        return R;
    }
    Eigen::MatrixXd KinematicTrajectoryOpti::create_Q(double w, const Eigen::MatrixXd &R) const
    {
        double Qt = w; // weight_t = 0.1
        Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx_, nx_);
        Q(0, 0) = Qt;
        Q.block(1, 1, R.rows(), R.cols()) = R;
        std::cout << "Q " << Q.rows() << " " << Q.cols() << std::endl;
        return Q;
    }
    Eigen::VectorXd KinematicTrajectoryOpti::create_Pr() const
    {
        // P_reference
        Eigen::VectorXd Pr = Eigen::VectorXd::Zero(n_ * dim_);
        std::cout << "Pr " << Pr.rows() << " " << Pr.cols() << std::endl;
        for (int i = 0; i < n_; i++)
            Pr.block(i * dim_, 0, dim_, 1) = control_points_[i];
        std::cout << Pr.transpose() << std::endl;
        return Pr;
    }
    Eigen::VectorXd KinematicTrajectoryOpti::create_q(const Eigen::VectorXd &Pr, const Eigen::MatrixXd &R) const
    {
        // q
        Eigen::MatrixXd q1 = Eigen::MatrixXd::Zero(1, 1);
        Eigen::MatrixXd q2 = -R * Pr;
        Eigen::VectorXd q = vstack(q1, q2).col(0);
        std::cout << "q " << q.size() << std::endl;
        // std::cout << q.transpose() << std::endl;
        return q;
    }
    Eigen::MatrixXd KinematicTrajectoryOpti::create_M() const
    {
        Eigen::MatrixXd M1_A = Eigen::MatrixXd::Identity(n_ - 1, n_ - 1);
        Eigen::MatrixXd M1 = kroneckerProduct_eye(M1_A, Eigen::MatrixXd::Identity(dim_, dim_));
        // std::cout << "M1 " << M1.rows() << " " << M1.cols() << std::endl;
        // std::cout << M1 << std::endl;

        Eigen::MatrixXd M2_A = subDiagIdentyMatrix(n_ - 1);
        Eigen::Matrix M2 = kroneckerProduct_subEye(M2_A, Eigen::MatrixXd::Identity(dim_, dim_));
        // std::cout << "M2 " << M2.rows() << " " << M2.cols() << std::endl;
        // std::cout << M2 << std::endl;

        Eigen::MatrixXd M3 = M1 - M2;
        // std::cout << "M3 " << M3.rows() << " " << M3.cols() << std::endl;
        // std::cout << M3 << std::endl;

        Eigen::MatrixXd M4 = Eigen::MatrixXd::Zero((n_ - 1) * dim_, dim_ + 1);
        M4.block(0, 1, dim_, dim_) = -1 * Eigen::MatrixXd::Identity(dim_, dim_);
        Eigen::MatrixXd M = hstack(M4, M3);
        std::cout << "M " << M.rows() << " " << M.cols() << std::endl;
        // std::cout << M << std::endl;
        return M;
    }
    Eigen::MatrixXd KinematicTrajectoryOpti::create_Aeq(const Eigen::MatrixXd &M) const
    {
        Eigen::MatrixXd Aq0 = hstack(Eigen::MatrixXd::Zero(dim_, 1), traj_.basis().Bt(dim_, 0));
        Eigen::MatrixXd Aq1 = hstack(Eigen::MatrixXd::Zero(dim_, 1), traj_.basis().Bt(dim_, 1));
        std::cout << "Aq0 " << Aq0.rows() << " " << Aq0.cols() << std::endl;
        // std::cout << Aq0 << std::endl;
        std::cout << "Aq1 " << Aq1.rows() << " " << Aq1.cols() << std::endl;
        // std::cout << Aq1 << std::endl;

        Eigen::MatrixXd Adq0 = traj_.basis().dBt_weighted(dim_, 0) * M;
        std::cout << "Adq0 " << Adq0.rows() << " " << Adq0.cols() << std::endl;
        // std::cout << Adq0 << std::endl;
        Eigen::MatrixXd Adq1 = traj_.basis().dBt_weighted(dim_, traj_.endTime()) * M;
        std::cout << "Adq1 " << Adq1.rows() << " " << Adq1.cols() << std::endl;
        // std::cout << Adq1 << std::endl;

        // Aq = [Aq0; Aq1; Adq0; Adq1]
        Eigen::MatrixXd Aq = vstack(Aq0, Aq1);
        Aq = vstack(Aq, Adq0);
        Aq = vstack(Aq, Adq1);
        std::cout << "Aq " << Aq.rows() << " " << Aq.cols() << std::endl;
        // std::cout << Aq << std::endl;
        return Aq;
    }
    Eigen::MatrixXd KinematicTrajectoryOpti::create_Lq() const
    {
        Eigen::MatrixXd Lq = vstack(p_start_.col(0), p_goal_.col(0));
        Lq = vstack(Lq, vel_start_.col(0));
        Lq = vstack(Lq, vel_goal_.col(0));
        std::cout << "Lq " << Lq.rows() << " " << Lq.cols() << std::endl;
        std::cout << Lq.transpose() << std::endl;
        return Lq;
    }
    Eigen::MatrixXd KinematicTrajectoryOpti::create_dM() const
    {
        Eigen::MatrixXd dM = Eigen::MatrixXd::Zero(1, nx_);
        dM(0, 0) = 1;
        std::cout << "dM " << dM.rows() << " " << dM.cols() << std::endl;
        return dM;
    }
    Eigen::MatrixXd KinematicTrajectoryOpti::create_dQmin() const
    {
        Eigen::VectorXd dQmin;
        for (int i = 0; i < n_inequ_constraint_; i++)
        {
            if (dQmin.cols() == 0)
                dQmin = vstack(vel_min_.col(0), vel_min_.col(0));
            else
                dQmin = vstack(dQmin.col(0), vel_min_.col(0));
        }
        std::cout << "dQmin " << dQmin.rows() << " " << dQmin.cols() << std::endl;
        std::cout << dQmin.transpose() << std::endl;
        return dQmin;
    }
    Eigen::MatrixXd KinematicTrajectoryOpti::create_dQmax() const
    {
        Eigen::VectorXd dQmax;
        for (int i = 0; i < n_inequ_constraint_; i++)
        {
            if (dQmax.cols() == 0)
                dQmax = vstack(vel_max_.col(0), vel_max_.col(0));
            else
                dQmax = vstack(dQmax.col(0), vel_max_.col(0));
        }
        std::cout << "dQmax " << dQmax.rows() << " " << dQmax.cols() << std::endl;
        std::cout << dQmax.transpose() << std::endl;
        return dQmax;
    }
    Eigen::MatrixXd KinematicTrajectoryOpti::create_Aieq(const Eigen::MatrixXd &M, const Eigen::MatrixXd &dM,
                                                         const Eigen::MatrixXd &dQmin, const Eigen::MatrixXd &dQmax) const
    {
        // dBWS = [dB_w(0), dB_w(0.25), dB_w(0.5), dB_w(0.75), dB_w(1)]
        // AiqL = dBWS * M - dQmin * dM >= 0
        // AiqU = dBWS * M - dQmax * dM <= 0
        Eigen::MatrixXd dBWS = dB_weighted_sequence();
        Eigen::MatrixXd AiqL = dBWS * M - dQmin * dM;
        Eigen::MatrixXd AiqU = dBWS * M - dQmax * dM;
        std::cout << "dQmin * dM " << (dQmin * dM).rows() << " " << (dQmin * dM).cols()
                  << std::endl;
        std::cout << "AiqL " << AiqL.rows() << " " << AiqL.cols() << std::endl;
        // std::cout << AiqL << std::endl;
        std::cout << "AiqU " << AiqU.rows() << " " << AiqU.cols() << std::endl;
        // std::cout << AiqU << std::endl;

        return vstack(AiqL, AiqU);
    }
    Eigen::VectorXd KinematicTrajectoryOpti::create_Lineq() const
    {
        double inf = std::numeric_limits<double>::infinity();
        int row = n_inequ_constraint_ * dim_;
        Eigen::VectorXd Lineq = Eigen::VectorXd::Zero(row);
        Lineq = vstack(Lineq, Eigen::VectorXd::Constant(row, -inf));
        std::cout << "Lineq " << Lineq.rows() << " " << Lineq.cols() << std::endl;
        std::cout << Lineq.transpose() << std::endl;
        return Lineq;
    }
    Eigen::VectorXd KinematicTrajectoryOpti::create_Uineq() const
    {
        double inf = std::numeric_limits<double>::infinity();
        int row = n_inequ_constraint_ * dim_;
        Eigen::VectorXd Uineq = Eigen::VectorXd::Constant(row, inf);
        Uineq = vstack(Uineq, Eigen::VectorXd::Zero(row));
        std::cout << "Uineq " << Uineq.rows() << " " << Uineq.cols() << std::endl;
        std::cout << Uineq.transpose() << std::endl;
        return Uineq;
    }
    BsplineTrajectory KinematicTrajectoryOpti::solve()
    {
        // init and config
        init(4, 10, control_points());
        traj_ = BsplineTrajectory(BsplineBasis(4, n_), control_points_);

        // formulate QP
        Eigen::MatrixXd R = create_R(3);
        Eigen::MatrixXd Q = create_Q(0.1, R);
        Eigen::VectorXd Pr = create_Pr();
        Eigen::VectorXd q = create_q(Pr, R);
        Eigen::MatrixXd M = create_M();
        Eigen::MatrixXd dM = create_dM();
        // equality constraint
        Eigen::MatrixXd Aeq = create_Aeq(M);
        Eigen::MatrixXd Lq = create_Lq();
        Eigen::MatrixXd Uq = Lq;
        // inequality constraint
        Eigen::MatrixXd dQmin = create_dQmin();
        Eigen::MatrixXd dQmax = create_dQmax();

        Eigen::MatrixXd Aieq = create_Aieq(M, dM, dQmin, dQmax);
        Eigen::MatrixXd A = vstack(Aeq, Aieq);

        Eigen::VectorXd Lineq = create_Lineq();
        Eigen::VectorXd Uineq = create_Uineq();
        Eigen::VectorXd L = vstack(Lq, Lineq);
        Eigen::VectorXd U = vstack(Uq, Uineq);

        // OSQP solver
        auto solution = OSQP_solver(Q, q, A, L, U);
        duration_ = solution.first;

        return BsplineTrajectory(BsplineBasis{traj_.basis().order(), n_, KnotVectorType::kClamptedUniform, 0, solution.first},
                                 solution.second);
    }
    std::vector<Eigen::VectorXd> KinematicTrajectoryOpti::solution_to_control_points(const OSQPFloat *x) const
    {
        std::vector<Eigen::VectorXd> control_points;
        for (int i = 0; i < n_; i++)
        {
            Eigen::VectorXd p = Eigen::VectorXd::Zero(dim_);
            for (int j = 0; j < dim_; j++)
                p(j) = x[i * dim_ + j + 1];
            control_points.push_back(p);
            std::cout << "p " << p.transpose() << std::endl;
        }
        return control_points;
    }
    std::pair<double, std::vector<Eigen::VectorXd>> KinematicTrajectoryOpti::OSQP_solver(const Eigen::MatrixXd &Q, const Eigen::VectorXd &q,
                                                                                         const Eigen::MatrixXd &A, const Eigen::VectorXd &L,
                                                                                         const Eigen::VectorXd &U)
    {
        Eigen::SparseMatrix<double> Q_sparse = Q.sparseView();
        std::shared_ptr<OSQPCscMatrix> Q_osqp = eigenSparseMatrixToCsc(Q_sparse);
        std::cout << "Q_osqp " << Q_osqp->nzmax << std::endl;
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
            osqp_setup(&solver, Q_osqp.get(), q_osqp, A_osqp.get(), L_osqp, U_osqp,
                       A.rows(), nx_, &settings);
        exitflag = osqp_solve(solver);
        // duration_ = solver->solution->x[0];
        std::cout << "solution: x ";
        for (int i = 0; i < nx_; i++)
            std::cout << solver->solution->x[i] << " ";
        std::cout << std::endl;
        return std::make_pair(solver->solution->x[0], solution_to_control_points(solver->solution->x));
    }
    void KinematicTrajectoryOpti::plot_traj()
    {
        BsplineTrajectory traj = solve();
        double dt = 1 / static_cast<double>(n_);
        std::vector<double> t0 = matplot::linspace(0, 1, n_);
        std::vector<double> t1 = matplot::linspace(0, 1, 100);
        std::vector<double> q0_v; // initial trajectory of dof(0)
        std::vector<double> q1_v; // initial trajectory of dof(1)
        std::vector<double> c0_v; // initial contorl point of dof(0)
        std::vector<double> c1_v; // initial contorl point of dof(1)

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
        matplot::subplot(3, 1, 0);
        matplot::plot(c0_v, c1_v, q0_v, q1_v, "--");
        matplot::hold(matplot::on);

        std::cout << "end time " << traj.endTime() << std::endl;
        std::vector<double> t2 = matplot::linspace(0, traj.endTime(), 100);
        std::vector<double> q2_v; // optimal trajectory of dof(0)
        std::vector<double> q3_v; // optimal trajectory of dof(1)
        std::vector<double> c2_v; // optimal control point of dof(0)
        std::vector<double> c3_v; // optimal control point of dof(1)
        for (int i = 0; i < t0.size(); i++)
        {
            c2_v.push_back(traj.controlPoints()[i](0));
            c3_v.push_back(traj.controlPoints()[i](1));
        }
        matplot::plot(c2_v, c3_v, "-*-");

        for (int i = 0; i < t2.size(); i++)
        {
            q2_v.push_back(traj.value(t2[i])(0));
            q3_v.push_back(traj.value(t2[i])(1));
        }
        std::cout << "q2_v " << q2_v.size() << " q3_v " << q3_v.size() << std::endl;
        matplot::plot(q2_v, q3_v, "--");
        matplot::grid(matplot::on);
        matplot::hold(matplot::off);
        matplot::subplot(3, 1, 1);
        std::vector<double> v0_v; // initial trajectory vel of dof(0)
        std::vector<double> v1_v; // initial trajectory vel of dof(1)
        std::vector<double> v2_v; // optimal trajectory vel of dof(0)
        std::vector<double> v3_v; // optimal trajectory vel of dof(1)
        for (int i = 0; i < t1.size(); i++)
        {
            v0_v.push_back(traj_.evalDerivative(t1[i])(0));
            v1_v.push_back(traj_.evalDerivative(t1[i])(1));
        }
        matplot::plot(t1, v0_v, t1, v1_v, "--");
        matplot::xlabel("Time(s)");
        matplot::ylabel("vel(rad/s)");
        matplot::grid(matplot::on);

        matplot::subplot(3, 1, 2);
        for (int i = 0; i < t2.size(); i++)
        {
            v2_v.push_back(traj.evalDerivative(t2[i])(0));
            v3_v.push_back(traj.evalDerivative(t2[i])(1));
        }
        matplot::plot(t2, v2_v, t2, v3_v, "--");
        matplot::xlabel("Time(s)");
        matplot::ylabel("opt_vel(rad/s)");
        matplot::grid(matplot::on);
        matplot::show();
    }

} // namespace gsmpl
