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
        KinematicTrajectoryOpti(const std::vector<double> &vel_min = {},
                                const std::vector<double> &vel_max = {},
                                const std::vector<double> &acc_min = {},
                                const std::vector<double> &acc_max = {},
                                int num = 10)
            : vel_min_(vel_min),
              vel_max_(vel_max),
              acc_min_(acc_min),
              acc_max_(acc_max),
              num_control_point_(num),
              dim_(7) {}

        const BsplineTrajectory &traj() const { return traj_; }

        BsplineTrajectory solve();
        void plot_traj();

    private:
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
        Eigen::MatrixXd Bt(double t)
        {
            assert(traj_.numControlPoints() == num_control_point_);

            Eigen::MatrixXd Bt =
                Eigen::MatrixXd::Zero(dim_, num_control_point_ * dim_);
            for (int i = 0; i < num_control_point_; i++)
                Bt.block(0, i * dim_, dim_, dim_) =
                    traj_.basis().evaluateBasisFunctionI(dim_, i, t).asDiagonal();
            std::cout << "Bt" << t << " " << Bt.rows() << " " << Bt.cols()
                      << std::endl;
            // std::cout << Bt << std::endl;
            return Bt;
        }

        Eigen::MatrixXd dBcoe(double t)
        {
            assert(traj_.numControlPoints() == num_control_point_);

            int n = num_control_point_ - 1;
            Eigen::MatrixXd dBcoe = Eigen::MatrixXd::Zero(dim_, n * dim_);
            for (int i = 0; i < n; i++)
                dBcoe.block(0, i * dim_, dim_, dim_) =
                    traj_.basis().dBcoe_I(dim_, i, t).asDiagonal();
            std::cout << "dBcoe" << t << " " << dBcoe.rows() << " " << dBcoe.cols()
                      << std::endl;
            // std::cout << dBcoe << std::endl;
            return dBcoe;
        }
        Eigen::MatrixXd dBcoe_sequence()
        {
            Eigen::MatrixXd out;

            traj_.basis().print_knots();
            double dt = 1.0 / static_cast<double>(num_vel_constraint_ - 1);

            Eigen::MatrixXd dBcoe_0 = dBcoe(0);
            Eigen::MatrixXd dBcoe_1 = dBcoe(dt);
            out = vstack(dBcoe_0, dBcoe_1);
            for (int i = 2; i < num_vel_constraint_; i++)
                out = vstack(out, dBcoe(dt * i));
            std::cout << "dBcoe_sequence " << out.rows() << " " << out.cols()
                      << std::endl;
            // std::cout << out << std::endl;
            return out;
        }
        std::vector<double> vel_min_;
        std::vector<double> vel_max_;
        std::vector<double> acc_min_;
        std::vector<double> acc_max_;
        std::vector<Eigen::VectorXd> control_points_;
        int num_control_point_;
        int num_vel_constraint_{5};
        int dim_;

        double duration_;
        BsplineTrajectory traj()
        {
            control_points_ = control_points();
            return BsplineTrajectory(BsplineBasis(4, num_control_point_),
                                     control_points_);
        }
        BsplineTrajectory traj_;
    };
} // namespace gsmpl
