#pragma once

#include <chrono>
#include <thread>

#include <assert.h>
#include <algorithm>
#include <stdexcept>

#include "bspline_trajectory.h"

namespace gsmpl
{
    BsplineTrajectory::BsplineTrajectory(
        BsplineBasis basis, std::vector<Eigen::VectorXd> control_points)
        : basis_(std::move(basis)), control_points_(std::move(control_points))
    {
        checkInvariants();
    }
    std::unique_ptr<Trajectory> BsplineTrajectory::clone() const
    {
        return std::make_unique<BsplineTrajectory>(*this);
    }

    Eigen::VectorXd BsplineTrajectory::value(const double time) const
    {
        return basis().evaluateCurve(controlPoints(),
                                     std::clamp(time, startTime(), endTime()));
    }
    bool BsplineTrajectory::doHasDerivative() const { return true; }
    Eigen::VectorXd BsplineTrajectory::doEvalDerivative(
        const double time, int derivative_order) const
    {
        if (derivative_order == 0)
        {
            return this->value(time);
        }
        else if (derivative_order >= basis_.order())
        {
            return Eigen::VectorXd::Zero(stateDof());
        }
        else if (derivative_order >= 1)
        {
            double clamped_time = std::clamp(time, startTime(), endTime());
            // For a bspline trajectory of order n, the evaluation of k th
            // derivative should take O(k^2) time by leveraging the sparsity of
            // basis value. This differs from DoMakeDerivative, which takes O(nk)
            // time.
            std::vector<double> derivative_knots(
                basis_.knots().begin() + derivative_order,
                basis_.knots().end() - derivative_order);
            BsplineBasis lower_order_basis =
                BsplineBasis(basis_.order() - derivative_order, derivative_knots);
            std::vector<Eigen::VectorXd> coefficients(controlPoints());
            std::vector<int> base_indices =
                basis_.computeActiveBasisFunctionIndices(clamped_time);
            for (int j = 1; j <= derivative_order; ++j)
            {
                for (int i = base_indices.front(); i <= base_indices.back() - j;
                     ++i)
                {
                    coefficients.at(i) = (basis_.order() - j) /
                                         (basis_.knots()[i + basis_.order()] -
                                          basis_.knots()[i + j]) *
                                         (coefficients[i + 1] - coefficients[i]);
                }
            }
            std::vector<Eigen::VectorXd> derivative_control_points(
                numControlPoints() - derivative_order,
                Eigen::VectorXd::Zero(stateDof()));
            for (int i : lower_order_basis.computeActiveBasisFunctionIndices(
                     clamped_time))
            {
                derivative_control_points.at(i) = coefficients.at(i);
            }
            return lower_order_basis.evaluateCurve(derivative_control_points,
                                                   clamped_time);
        }
        else
        {
            throw std::logic_error(
                "Invalid derivative order ({}). The derivative order must "
                "be greater than or equal to 0.");
        }
    }

    std::unique_ptr<Trajectory> BsplineTrajectory::doMakeDerivative(
        int derivative_order) const
    {
        if (derivative_order == 0)
        {
            return this->clone();
        }
        else if (derivative_order > basis_.degree())
        {
            std::vector<double> derivative_knots;
            derivative_knots.push_back(basis_.knots().front());
            derivative_knots.push_back(basis_.knots().back());
            std::vector<Eigen::VectorXd> control_points(
                1, Eigen::VectorXd::Zero(stateDof()));
            return std::make_unique<BsplineTrajectory>(
                BsplineBasis(1, derivative_knots), control_points);
        }
        else if (derivative_order > 1)
        {
            return this->makeDerivative(1)->makeDerivative(derivative_order - 1);
        }
        else if (derivative_order == 1)
        {
            std::vector<double> derivative_knots;
            const int num_derivative_knots = basis_.knots().size() - 2;
            derivative_knots.reserve(num_derivative_knots);
            for (int i = 1; i <= num_derivative_knots; ++i)
            {
                derivative_knots.push_back(basis_.knots()[i]);
            }
            std::vector<Eigen::VectorXd> derivative_control_points;
            derivative_control_points.reserve(numControlPoints() - 1);
            for (int i = 0; i < numControlPoints() - 1; ++i)
            {
                derivative_control_points.push_back(
                    static_cast<double>(basis_.degree()) /
                    (basis_.knots()[i + basis_.order()] - basis_.knots()[i + 1]) *
                    (controlPoints()[i + 1] - controlPoints()[i]));
            }
            return std::make_unique<BsplineTrajectory>(
                BsplineBasis(basis_.order() - 1, derivative_knots),
                derivative_control_points);
        }
        else
        {
            throw std::logic_error(
                "Invalid derivative order ({}). The derivative order must "
                "be greater than or equal to 0.");
        }
    }
    void BsplineTrajectory::checkInvariants() const
    {
        assert(static_cast<int>(control_points_.size()) ==
               basis_.numBasisFunctions());
    }

    double BsplineKinematicConstraintTraj::scale()
    {
        double s_vel = 1;
        double s_acc = 1;
        double dt = (traj_.endTime() - traj_.startTime()) /
                    static_cast<double>(num_waypoints_);

        for (int i = 0; i < num_waypoints_; i++)
        {
            Eigen::VectorXd position = traj_.value(dt * i);
            Eigen::VectorXd vel = traj_.evalDerivative(dt * i, 1);
            Eigen::VectorXd acc = traj_.evalDerivative(dt * i, 2);
            assert(position.size() == dim_);
            assert(vel.size() == dim_);
            assert(acc.size() == dim_);

            for (int j = 0; j < dim_; j++)
            {
                s_vel =
                    std::min(jointScale(vel_min_[j], vel_max_[j], vel(j)), s_vel);
                s_acc =
                    std::min(jointScale(acc_min_[j], acc_max_[j], vel(j)), s_acc);
            }
        }
        return std::min(s_vel, s_acc);
    }

    double BsplineKinematicConstraintTraj::jointScale(double l, double u,
                                                      double value)
    {
        assert(l <= u);
        assert(u > 0);
        assert(l < 0);
        if (value < l)
            return std::abs(l / value);
        if (value > u)
            return std::abs(u / value);
        return 1;
    }
} // namespace gsmpl
