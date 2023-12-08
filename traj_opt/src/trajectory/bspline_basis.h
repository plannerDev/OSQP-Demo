#pragma once

#include <vector>
#include <iostream>
#include <array>
#include <assert.h>
#include <Eigen/Core>

namespace gsmpl {
enum class KnotVectorType { kUniform, kClamptedUniform };

class BsplineBasis final {
    using StatePosition = Eigen::VectorXd;

public:
    BsplineBasis(int order = 0, std::vector<double> knots = {});
    BsplineBasis(int order, int num_basis_functions,
                 KnotVectorType type = KnotVectorType::kClamptedUniform,
                 const double initial_parameter_value = 0,
                 const double final_parameter_value = 1);

    int order() const { return order_; }
    int degree() const { return order() - 1; }
    int numBasisFunctions() const { return knots_.size() - order_; }
    const std::vector<double>& knots() const { return knots_; }
    double initialParameterValue() const { return knots()[order() - 1]; }
    double finalParameterValue() const { return knots()[numBasisFunctions()]; }

    int findContainingInterval(const double parameter_value) const;
    std::vector<int> computeActiveBasisFunctionIndices(
        const std::array<double, 2>& parameter_interval) const;

    std::vector<int> computeActiveBasisFunctionIndices(
        const double parameter_value) const;

    StatePosition evaluateCurve(
        const std::vector<StatePosition>& control_points,
        const double parameter_value) const;
    Eigen::VectorXd evaluateBasisFunctionI(int state_dim, int index,
                                           double parameter_value) const;
    Eigen::VectorXd dBcoe_I(int state_dim, int index,
                            double parameter_value) const;

    void print_knots() const {
        std::cout << "knots_ ";
        for (int i = 0; i < knots_.size(); i++)
            std::cout << knots_[i] << " ";
        std::cout << std::endl;
    }

private:
    bool CheckInvariants() const;

    int order_;
    std::vector<double> knots_;
};

} // namespace gsmpl
