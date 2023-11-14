#include <memory>
#include <assert.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <osqp/osqp.h>

extern "C"
{
    extern OSQPCscMatrix *csc_spalloc(OSQPInt m, OSQPInt n, OSQPInt nzmax, OSQPInt values, OSQPInt triplet);
    extern void csc_spfree(OSQPCscMatrix *A);
}

std::shared_ptr<OSQPCscMatrix> eiegnSparseMatrixToCsc(Eigen::SparseMatrix<double> &sparse_matrix);

Eigen::SparseMatrix<double> cscToEiegnSparseMatrix(const std::shared_ptr<OSQPCscMatrix> &csc);

// need to assgin array @param a size
void matrixToArray(const Eigen::MatrixXd &m, OSQPFloat *a);

OSQPInt update_data_mat(OSQPSolver *solver,
                        Eigen::SparseMatrix<double> &Px_new,
                        Eigen::SparseMatrix<double> &Ax_new,
                        std::shared_ptr<OSQPCscMatrix> &P_csc,
                        std::shared_ptr<OSQPCscMatrix> &A_csc);

void printCsc(const std::shared_ptr<OSQPCscMatrix> csc, const std::string &desc = "");
void printEigenSparseMatrix(const Eigen::SparseMatrix<double> &A, const std::string &desc = "");
void test();

class MPC
{
public:
    MPC();
    int solve();

    OSQPFloat inf = std::numeric_limits<OSQPFloat>::infinity();

private:
    Eigen::SparseMatrix<double> create_Ad();
    Eigen::SparseMatrix<double> create_Bd();
    Eigen::VectorXd create_xmin();
    Eigen::VectorXd create_xmax();
    Eigen::VectorXd create_umin();
    Eigen::VectorXd create_umax();
    Eigen::VectorXd create_x0();
    Eigen::VectorXd create_xr();
    Eigen::MatrixXd create_Q();
    Eigen::MatrixXd create_QN();
    Eigen::MatrixXd create_R();
    Eigen::MatrixXd create_P();
    Eigen::MatrixXd create_q();
    Eigen::MatrixXd create_Ax();
    Eigen::MatrixXd create_Bu();
    Eigen::MatrixXd create_Aeq(const Eigen::MatrixXd &Ax, const Eigen::MatrixXd &Bu);
    Eigen::MatrixXd create_leq();
    Eigen::MatrixXd create_ueq();
    Eigen::MatrixXd create_Aineq();
    Eigen::MatrixXd create_lineq();
    Eigen::MatrixXd create_uineq();

    const int nx_{12};
    const int nu_{4};

    Eigen::SparseMatrix<double> Ad_;
    Eigen::SparseMatrix<double> Bd_;

    Eigen::MatrixXd Q_;
    Eigen::MatrixXd QN_;
    Eigen::MatrixXd R_;

    Eigen::VectorXd xmin_;
    Eigen::VectorXd xmax_;
    Eigen::VectorXd umin_;
    Eigen::VectorXd umax_;

    Eigen::VectorXd x0_;
    Eigen::VectorXd xr_;

    const int N_;
    const int nsim_;
    const int nVariable_; // Variable num
};

class TestCase
{
public:
    int setupAndSolve();
    int updateVectors();
    int updateMatrix();
    int mpcSolver()
    {
        MPC mpc;
        return mpc.solve();
    }
};
