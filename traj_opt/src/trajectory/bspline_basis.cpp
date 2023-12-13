#pragma once

#include <assert.h>
#include <stdexcept>
#include "bspline_basis.h"

namespace gsmpl
{
    namespace
    {
        std::vector<double> makeKnotVector(int order, int num_basis_functions,
                                           KnotVectorType type,
                                           const double initial_parameter_value,
                                           const double final_parameter_value)
        {
            if (num_basis_functions < order)
            {
                throw std::logic_error(
                    "The number of basis functions ({}) should be greater than or "
                    "equal to the order ({}).");
            }
            assert(initial_parameter_value <= final_parameter_value);
            const int num_knots{num_basis_functions + order};
            std::vector<double> knots(num_knots);
            const double knot_interval =
                (final_parameter_value - initial_parameter_value) /
                (num_basis_functions - order + 1.0);
            for (int i = 0; i < num_knots; ++i)
            {
                if (i < order && type == KnotVectorType::kClamptedUniform)
                {
                    knots.at(i) = initial_parameter_value;
                }
                else if (i >= num_basis_functions &&
                         type == KnotVectorType::kClamptedUniform)
                {
                    knots.at(i) = final_parameter_value;
                }
                else
                {
                    knots.at(i) =
                        initial_parameter_value + knot_interval * (i - (order - 1));
                }
            }
            return knots;
        }

        bool less_than_with_cast(const double val, const double other)
        {
            return static_cast<bool>(val < other);
        }
    } // namespace
    BsplineBasis::BsplineBasis(int order, std::vector<double> knots)
        : order_(order), knots_(std::move(knots))
    {
        if (static_cast<int>(knots_.size()) < 2 * order)
        {
            throw std::logic_error(
                "The number of knots ({}) should be greater than or "
                "equal to twice the order ({}).");
        }
        assert(CheckInvariants());
    }
    BsplineBasis::BsplineBasis(int order, int num_basis_functions,
                               KnotVectorType type,
                               const double initial_parameter_value,
                               const double final_parameter_value)
        : BsplineBasis(order, makeKnotVector(order, num_basis_functions, type,
                                             initial_parameter_value,
                                             final_parameter_value)) {}

    int BsplineBasis::findContainingInterval(const double parameter_value) const
    {
        assert(parameter_value >= initialParameterValue());
        assert(parameter_value <= finalParameterValue());
        const std::vector<double> &t = knots();
        const double t_bar = parameter_value;
        return std::distance(
            t.begin(), std::prev(t_bar < finalParameterValue()
                                     ? std::upper_bound(t.begin(), t.end(), t_bar,
                                                        less_than_with_cast)
                                     : std::lower_bound(t.begin(), t.end(), t_bar,
                                                        less_than_with_cast)));
    }
    std::vector<int> BsplineBasis::computeActiveBasisFunctionIndices(
        const std::array<double, 2> &parameter_interval) const
    {
        assert(parameter_interval[0] <= parameter_interval[1]);
        assert(parameter_interval[0] >= initialParameterValue());
        assert(parameter_interval[1] <= finalParameterValue());
        const int first_active_index =
            findContainingInterval(parameter_interval[0]) - order() + 1;
        const int final_active_index =
            findContainingInterval(parameter_interval[1]);
        std::vector<int> active_control_point_indices{};
        active_control_point_indices.reserve(final_active_index -
                                             first_active_index);
        for (int i = first_active_index; i <= final_active_index; ++i)
        {
            active_control_point_indices.push_back(i);
        }
        return active_control_point_indices;
    }

    std::vector<int> BsplineBasis::computeActiveBasisFunctionIndices(
        const double parameter_value) const
    {
        return computeActiveBasisFunctionIndices(
            {parameter_value, parameter_value});
    }
    // B(i)(t)
    Eigen::VectorXd BsplineBasis::Bit(int dim, int index, double t) const
    {
        assert(index < numBasisFunctions());
        Eigen::VectorXd s = Eigen::VectorXd::Zero(dim);
        std::vector<Eigen::VectorXd> delta(numBasisFunctions(), s);
        for (int i = 0; i < dim; i++)
            delta[index](i) = 1.0;
        return evaluateCurve(delta, t);
    }
    Eigen::MatrixXd BsplineBasis::Bt(int dim, double t) const
    {
        int n = numBasisFunctions();
        Eigen::MatrixXd Bt = Eigen::MatrixXd::Zero(dim, n * dim);
        for (int i = 0; i < n; i++)
            Bt.block(0, i * dim, dim, dim) = Bit(dim, i, t).asDiagonal();
        std::cout << "Bt" << t << " " << Bt.rows() << " " << Bt.cols() << std::endl;
        // std::cout << Bt << std::endl;
        return Bt;
    }
    // dot_Bi_coefficient
    // dBcoe_{i}(t) = \frac{p}{t_{i + p + 1} - t_{i + 1}} * B_{i + 1}^{p - 1}(t)
    Eigen::VectorXd BsplineBasis::dBit_weighted(int dim, int index, double t) const
    {
        assert(index < numBasisFunctions() - 1);
        double p = static_cast<double>(degree());
        double dt = knots()[index + p + 1] - knots()[index + 1];
        if (degree() == 0 || dt == 0)
            return Eigen::VectorXd::Zero(dim);

        return (p / dt) * dBit(dim, index, t);
    }
    Eigen::MatrixXd BsplineBasis::dBt_weighted(int dim, double t) const
    {
        int n = numBasisFunctions() - 1;
        Eigen::MatrixXd dBtcoe = Eigen::MatrixXd::Zero(dim, n * dim);
        for (int i = 0; i < n; i++)
            dBtcoe.block(0, i * dim, dim, dim) =
                dBit_weighted(dim, i, t).asDiagonal();
        std::cout << "dBcoe " << t << " " << dBtcoe.rows() << " " << dBtcoe.cols()
                  << std::endl;
        std::cout << dBtcoe << std::endl;
        return dBtcoe;
    }
    Eigen::VectorXd BsplineBasis::dBit(int dim, int index, double t) const
    {
        std::vector<double> derivative_knots;
        const int num_derivative_knots = knots().size() - 2;
        derivative_knots.reserve(num_derivative_knots);
        for (int i = 1; i <= num_derivative_knots; ++i)
            derivative_knots.push_back(knots()[i]);

        BsplineBasis dot_basis(order() - 1, derivative_knots);

        return dot_basis.Bit(dim, index, t);
    }
    Eigen::VectorXd BsplineBasis::evaluateCurve(
        const std::vector<Eigen::VectorXd> &control_points,
        const double parameter_value) const
    {
        assert(static_cast<int>(control_points.size()) == numBasisFunctions());
        assert(parameter_value >= initialParameterValue());
        assert(parameter_value <= finalParameterValue());

        // Define short names to match notation in [1].
        const std::vector<double> &t = knots();
        const double t_bar = parameter_value;
        const int k = order();

        /* Find the index, 𝑙, of the greatest knot that is less than or equal to
        t_bar and strictly less than finalParameterValue(). */
        const int ell = findContainingInterval(t_bar);
        // The vector that stores the intermediate de Boor points (the pᵢʲ in
        // [1]).
        std::vector<StatePosition> p(order());
        /* For j = 0, i goes from ell down to ell - (k - 1). Define r such that
        i = ell - r. */
        for (int r = 0; r < k; ++r)
        {
            const int i = ell - r;
            p.at(r) = control_points.at(i);
        }
        /* For j = 1, ..., k - 1, i goes from ell down to ell - (k - j - 1).
        Again, i = ell - r. */
        for (int j = 1; j < k; ++j)
        {
            for (int r = 0; r < k - j; ++r)
            {
                const int i = ell - r;
                // α = (t_bar - t[i]) / (t[i + k - j] - t[i]);
                const double alpha =
                    (t_bar - t.at(i)) / (t.at(i + k - j) - t.at(i));
                p.at(r) = (1.0 - alpha) * p.at(r + 1) + alpha * p.at(r);
            }
        }
        return p.front();
    }
    void BsplineBasis::print_knots() const
    {
        std::cout << "knots ";
        for (int i = 0; i < knots_.size(); i++)
            std::cout << knots_[i] << " ";
        std::cout << std::endl;
    }

    bool BsplineBasis::CheckInvariants() const
    {
        return std::is_sorted(knots_.begin(), knots_.end(), less_than_with_cast) &&
               static_cast<int>(knots_.size()) >= 2 * order_;
    }
} // namespace gsmpl
