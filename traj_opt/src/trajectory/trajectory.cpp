#pragma once

#include <stdexcept>
#include "trajectory.h"

namespace gsmpl
{
    Eigen::VectorXd Trajectory::doEvalDerivative(const double t,
                                                 int derivative_order) const
    {
        if (hasDerivative())
        {
            throw std::logic_error(
                "Trajectory classes that promise derivatives via "
                "do_has_derivative() "
                "must implement DoEvalDerivative().");
        }
        else
        {
            throw std::logic_error(
                "You asked for derivatives from a class that does not support "
                "derivatives.");
        }
    }

    std::unique_ptr<Trajectory> Trajectory::doMakeDerivative(
        int derivative_order) const
    {
        if (hasDerivative())
        {
            throw std::logic_error(
                "Trajectory classes that promise derivatives via "
                "do_has_derivative() "
                "must implement DoMakeDerivative().");
        }
        else
        {
            throw std::logic_error(
                "You asked for derivatives from a class that does not support "
                "derivatives.");
        }
    }
} // namespace gsmpl
