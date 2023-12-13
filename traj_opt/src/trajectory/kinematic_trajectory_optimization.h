#pragma once

#include <Eigen/Core>
#include <osqp/osqp.h>
#include "math_utility.h"
#include "bspline_trajectory.h"
#include "bspline_basis.h"

namespace gsmpl
{
    std::shared_ptr<OSQPCscMatrix> eigenSparseMatrixToCsc(
        Eigen::SparseMatrix<double> &A);

    Eigen::SparseMatrix<double> cscToEigenSparseMatrix(
        const std::shared_ptr<OSQPCscMatrix> &csc);

    // need to assgin array @param a size
    void matrixToArray(const Eigen::MatrixXd &m, OSQPFloat *a);

    OSQPInt update_data_mat(OSQPSolver *solver, Eigen::SparseMatrix<double> &Px_new,
                            Eigen::SparseMatrix<double> &Ax_new,
                            std::shared_ptr<OSQPCscMatrix> &P_csc,
                            std::shared_ptr<OSQPCscMatrix> &A_csc);

    void printCsc(const std::shared_ptr<OSQPCscMatrix> csc,
                  const std::string &desc = "");
    void printEigenSparseMatrix(const Eigen::SparseMatrix<double> &A,
                                const std::string &desc = "");
    class KinematicTrajectoryOpti
    {
    public:
        KinematicTrajectoryOpti() {}

        void init(int n_equa_constraint, int n_inequ_constraint, const std::vector<Eigen::VectorXd> &contorl_points);
        const BsplineTrajectory &traj() const { return traj_; }

        BsplineTrajectory solve();
        void plot_traj();

    private:
        Eigen::MatrixXd create_R(double w) const;
        Eigen::MatrixXd create_Q(double w, const Eigen::MatrixXd &R) const;
        Eigen::VectorXd create_Pr() const;
        Eigen::VectorXd create_q(const Eigen::VectorXd &Pr, const Eigen::MatrixXd &R) const;
        Eigen::MatrixXd create_M() const;
        Eigen::MatrixXd create_Aeq(const Eigen::MatrixXd &M) const;
        Eigen::MatrixXd create_Lq() const;
        Eigen::MatrixXd create_dM() const;
        Eigen::MatrixXd create_dQmin() const;
        Eigen::MatrixXd create_dQmax() const;
        Eigen::MatrixXd create_Aieq(const Eigen::MatrixXd &M, const Eigen::MatrixXd &dM,
                                    const Eigen::MatrixXd &dQmin, const Eigen::MatrixXd &dQmax) const;
        Eigen::VectorXd create_Lineq() const;
        Eigen::VectorXd create_Uineq() const;

        std::vector<Eigen::VectorXd> OSQP_solver(const Eigen::MatrixXd &Q, const Eigen::VectorXd &q,
                                                 const Eigen::MatrixXd &A, const Eigen::VectorXd &L,
                                                 const Eigen::VectorXd &U);
        std::vector<Eigen::VectorXd> solution_to_control_points(const OSQPFloat *x) const;

        // dBWSequence = [dBWeighted(t_0), ... , dBWeighted(t_m)]
        Eigen::MatrixXd dB_weighted_sequence() const;

        int dim_;
        int n_;                  // number of control points
        int nx_;                 // number of variables nx = n * dim + 1
        int nc_;                 // number of constraints
        int n_equa_constraint_;  // number of equality constraints
        int n_inequ_constraint_; // number of inequality constraints
        int num_vel_constraint_{10};
        std::vector<Eigen::VectorXd> control_points_;

        Eigen::VectorXd p_start_;
        Eigen::VectorXd p_goal_;
        Eigen::VectorXd vel_start_;
        Eigen::VectorXd vel_goal_;

        Eigen::VectorXd vel_min_;
        Eigen::VectorXd vel_max_;
        Eigen::VectorXd acc_min_;
        Eigen::VectorXd acc_max_;

        double duration_;
        BsplineTrajectory traj_;
    };
} // namespace gsmpl
