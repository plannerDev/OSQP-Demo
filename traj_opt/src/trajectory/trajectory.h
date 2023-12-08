#pragma once

#include <vector>
#include <memory>
#include <Eigen/Core>

namespace gsmpl {
class Trajectory {
public:
    Trajectory() = default;
    virtual ~Trajectory() = default;

    virtual std::unique_ptr<Trajectory> clone() const = 0;

    virtual std::size_t stateDof() const = 0;
    virtual double startTime() const = 0;
    virtual double endTime() const = 0;

    virtual Eigen::VectorXd value(const double t) const = 0;

    bool hasDerivative() const { return doHasDerivative(); }
    Eigen::VectorXd evalDerivative(const double t,
                                   int derivative_order = 1) const {
        return doEvalDerivative(t, derivative_order);
    }
    std::unique_ptr<Trajectory> makeDerivative(int derivative_order = 1) const {
        return doMakeDerivative(derivative_order);
    }

protected:
    virtual bool doHasDerivative() const { return false; }
    virtual Eigen::VectorXd doEvalDerivative(const double t,
                                             int derivative_order) const;
    virtual std::unique_ptr<Trajectory> doMakeDerivative(
        int derivative_order) const;
};
} // namespace gsmpl
