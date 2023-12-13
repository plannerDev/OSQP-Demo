#pragma once

#include <math.h>
#include <Eigen/Core>
#include "trajectory.h"
#include "bspline_basis.h"

namespace gsmpl
{

    class BsplineTrajectory final : public Trajectory
    {
    public:
        BsplineTrajectory(BsplineBasis basis,
                          std::vector<Eigen::VectorXd> control_points);
        BsplineTrajectory() = default;
        ~BsplineTrajectory() = default;

        std::unique_ptr<Trajectory> clone() const override;
        Eigen::VectorXd value(const double time) const override;

        std::size_t stateDof() const override { return controlPoints()[0].size(); }
        double startTime() const override { return basis_.t0(); }
        double endTime() const override { return basis_.tf(); }
        int numControlPoints() const { return basis_.numBasisFunctions(); }
        const std::vector<Eigen::VectorXd> &controlPoints() const
        {
            return control_points_;
        }
        const BsplineBasis &basis() const { return basis_; }

    private:
        bool doHasDerivative() const override;
        Eigen::VectorXd doEvalDerivative(const double t,
                                         int derivative_order) const override;
        std::unique_ptr<Trajectory> doMakeDerivative(
            int derivative_order) const override;
        void checkInvariants() const;

        BsplineBasis basis_;
        std::vector<Eigen::VectorXd> control_points_;
    };

    class BsplineKinematicConstraintTraj
    {
    public:
        BsplineKinematicConstraintTraj(const std::vector<double> &vel_min,
                                       const std::vector<double> &vel_max,
                                       const std::vector<double> &acc_min,
                                       const std::vector<double> &acc_max,
                                       int num = 100)
            : vel_min_(vel_min),
              vel_max_(vel_max),
              acc_min_(acc_min),
              acc_max_(acc_max),
              num_waypoints_(num),
              dim_(vel_min.size())
        {
            duration_ = scale();
            traj_ = BsplineTrajectory(
                BsplineBasis(traj_.basis().order(),
                             traj_.basis().numBasisFunctions(),
                             KnotVectorType::kClamptedUniform, 0, duration_),
                traj_.controlPoints());
        }

        const BsplineTrajectory &traj() const { return traj_; }

    private:
        double scale();
        double jointScale(double l, double u, double value);

        std::vector<double> vel_min_;
        std::vector<double> vel_max_;
        std::vector<double> acc_min_;
        std::vector<double> acc_max_;
        int num_waypoints_;
        int dim_;

        double duration_;
        BsplineTrajectory traj_;
    };
} // namespace gsmpl
