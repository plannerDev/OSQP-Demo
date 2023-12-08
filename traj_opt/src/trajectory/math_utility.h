#pragma once

#include <iostream>
#include <limits>
#include <random>
#include <Eigen/Geometry>
#include <Eigen/SparseCore>
#ifdef _WIN32
#include <corecrt_math_defines.h>
#define _USE_MATH_DEFINES
#endif

namespace gsmpl {
Eigen::Isometry3d rotateZAxis(double theta);
Eigen::Isometry3d rotateThetaAxis(double theta, const Eigen::Vector3d &axis);

static double sign(double x) {
    if (x < -std::numeric_limits<double>::min())
        return -1.0;
    if (x > std::numeric_limits<double>::min())
        return 1.0;
    else
        return 0;
}

double normalizeAngle(double theta);  // [-M_PI, M_PI]
double mappingZeroToPI(double theta); // [0, M_PI]

double unitNBallMeasure(std::size_t N, double r = 1.0);
double phsMeasure(std::size_t N, double minCost, double cost);

/*
  kroneckerProduct:
  A ⊗ B = | a11*B  a12*B  ...  a1n*B |
          | a21*B  a22*B  ...  a2n*B |
          | ...    ...    ...  ...   |
          | am1*B  am2*B  ...  amn*B |
  A = Eigen::MatrixXd::Identity();
  A = | B 0 0 0 |
      | 0 B 0 0 |
      | 0 0 B 0 |
      | 0 0 0 B |
*/
Eigen::MatrixXd kroneckerProduct_eye(const Eigen::MatrixXd &lhs,
                                     const Eigen::MatrixXd &rhs);
/*
  kroneckerProduct:
  A ⊗ B = | a11*B  a12*B  ...  a1n*B |
          | a21*B  a22*B  ...  a2n*B |
          | ...    ...    ...  ...   |
          | am1*B  am2*B  ...  amn*B |
  A = subDiagIdentyMatrix();
  A = | 0 0 0 0 |
      | B 0 0 0 |
      | 0 B 0 0 |
      | 0 0 B 0 |
      | 0 0 0 B |
*/
Eigen::MatrixXd kroneckerProduct_subEye(const Eigen::MatrixXd &A,
                                        const Eigen::MatrixXd &B);

/*
  subDiagIdentyMatrix:
  | 0 0 0 0 |
  | 1 0 0 0 |
  | 0 1 0 0 |
  | 0 0 1 0 |
  | 0 0 0 1 |
*/
Eigen::SparseMatrix<double> subDiagIdentyMatrix(int n);
// [A B]
Eigen::MatrixXd hstack(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B);

// [A; B]
Eigen::MatrixXd vstack(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B);

class RNG {
public:
    /* Generate a random real within given bounds: [\e lower_bound, \e
     * upper_bound) */
    double uniformReal(double lowerBound, double upperBound) {
        assert(lowerBound <= upperBound);
        std::random_device rd;
        std::mt19937 gen(rd());
        return (upperBound - lowerBound) * uni_dist_(gen) + lowerBound;
    }
    double uniformReal01() {
        std::random_device rd;
        std::mt19937 gen(rd());
        return uni_dist_(gen);
    }

    // [lowerBound, upperBound]
    int uniformInt(int lowerBound, int upperBound) {
        assert(lowerBound <= upperBound);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distri(lowerBound, upperBound);
        return distri(gen);
    }

    std::vector<double> uniformUnitSphere(std::size_t dim) {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::vector<double> v;
        double norm = 0.0;
        for (std::size_t i = 0; i < dim; i++) {
            double x = uni_dist_(gen) - 0.5;
            v.push_back(x);
            norm += x * x;
        }
        norm = std::sqrt(norm);

        for (std::size_t i = 0; i < dim; i++)
            v[i] = v[i] / norm;

        return v;
    }

    std::vector<double> uniformInBall(std::size_t dim, double r) {
        std::vector<double> sphere = uniformUnitSphere(dim);
        double radiusScale =
            r * std::pow(uniformReal01(), 1.0 / static_cast<double>(dim));
        std::vector<double> ball;

        for (const auto &p : sphere)
            ball.push_back(radiusScale * p);
        return ball;
    }

private:
    std::uniform_real_distribution<> uni_dist_{
        0.0, std::nextafter(1.0, std::numeric_limits<double>::max())}; // [0, 1]
};
} // namespace gsmpl
