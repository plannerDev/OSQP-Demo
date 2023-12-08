#pragma once

#include <cmath>
#include "math_utility.h"

namespace gsmpl
{
    Eigen::Isometry3d rotateZAxis(double theta)
    {
        // theta radian
        Eigen::AngleAxisd angleAxis(theta, Eigen::Vector3d(0, 0, 1));
        Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
        tf.rotate(angleAxis);
        return tf;
    }

    Eigen::Isometry3d rotateThetaAxis(double theta, const Eigen::Vector3d &axis)
    {
        // theta radian
        Eigen::AngleAxisd angleAxis(theta, axis);
        Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
        tf.rotate(angleAxis);
        return tf;
    }

    // [-M_PI, M_PI]
    double normalizeAngle(double theta)
    {
        double a = fmod(theta, 2.0 * M_PI);
        if ((-M_PI <= a) && (a <= M_PI))
            return a;
        if (a < -M_PI)
            return a + 2 * M_PI;
        else
            return a - 2 * M_PI;
    }

    double mappingZeroToPI(double theta)
    {
        double a = normalizeAngle(theta);
        if ((0 <= a) && (a <= M_PI))
            return a;
        else
            return a + M_PI;
    }

    double unitNBallMeasure(std::size_t N, double r)
    {
        double n = static_cast<double>(N);
        return std::pow(std::sqrt(M_PI) * r, n) / std::tgamma(n * 0.5 + 1.0);
    }

    double phsMeasure(std::size_t N, double minCost, double cost)
    {
        assert(minCost <= cost);
        double conjugateDiameter = std::sqrt(cost * cost - minCost * minCost);
        double lebsegueMeasure = cost * 0.5;
        for (std::size_t i = 0; i < N - 1; i++)
            lebsegueMeasure = lebsegueMeasure * conjugateDiameter * 0.5;
        lebsegueMeasure = lebsegueMeasure * unitNBallMeasure(N);
        return lebsegueMeasure;
    }

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
                                         const Eigen::MatrixXd &rhs)
    {
        Eigen::MatrixXd out(lhs.rows() * rhs.rows(), lhs.cols() * rhs.cols());
        out.setZero();
        for (int i = 0; i < lhs.rows(); i++)
        {
            out.block(i * rhs.rows(), i * rhs.cols(), rhs.rows(), rhs.cols()) =
                lhs(i, i) * rhs;
        }
        // std::cout << "kroneckerProduct_eye size " << out.rows() << " " <<
        // out.cols() << "\n"
        //           << out << std::endl;
        return out;
    }
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
                                            const Eigen::MatrixXd &B)
    {
        Eigen::MatrixXd out(A.rows() * B.rows(), A.cols() * B.cols());
        out.setZero();
        for (int i = 1; i < A.rows(); i++)
        {
            out.block(i * B.rows(), (i - 1) * B.cols(), B.rows(), B.cols()) =
                A(i, i - 1) * B;
        }
        return out;
    }
    /*
      subDiagIdentyMatrix:
      | 0 0 0 0 |
      | 1 0 0 0 |
      | 0 1 0 0 |
      | 0 0 1 0 |
      | 0 0 0 1 |
    */
    Eigen::SparseMatrix<double> subDiagIdentyMatrix(int n)
    {
        Eigen::SparseMatrix<double> out(n, n);
        for (int i = 1; i < n; i++)
            out.insert(i, i - 1) = 1.0;

        out.finalize();

        // std::cout << "subDiagIdentyMatrix \n"
        //           << out.toDense() << std::endl;
        return out;
    }
    // [A B]
    Eigen::MatrixXd hstack(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
    {
        assert(A.rows() == B.rows());
        Eigen::MatrixXd out(A.rows(), A.cols() + B.cols());
        out.setZero();
        out.block(0, 0, A.rows(), A.cols()) = A;
        out.block(0, A.cols(), B.rows(), B.cols()) = B;
        // std::cout << "hstack size " << out.rows() << " " << out.cols() << "\n"
        //           << out << std::endl;
        return out;
    }

    // [A; B]
    Eigen::MatrixXd vstack(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
    {
        assert(A.cols() == B.cols());
        Eigen::MatrixXd out(A.rows() + B.rows(), A.cols());
        out.setZero();
        out.block(0, 0, A.rows(), A.cols()) = A;
        out.block(A.rows(), 0, B.rows(), B.cols()) = B;
        // std::cout << "vstack size " << out.rows() << " " << out.cols() << "\n"
        //           << out << std::endl;
        return out;
    }
} // namespace gsmpl
