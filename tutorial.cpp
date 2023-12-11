// A simple program that computes the square root of a number
#include <iostream>
#include <string>
#include <matplot/matplot.h>

#include "traj_opt/src/test/osqp_demo.h"
#include "traj_opt/src/trajectory/kinematic_trajectory_optimization.h"

int plot_test()
{
  std::vector<double> x = matplot::linspace(0, 2 * matplot::pi);
  std::vector<double> y = matplot::transform(x, [](auto x)
                                             { return sin(x); });

  matplot::plot(x, y, "-o");
  matplot::hold(matplot::on);
  matplot::plot(x, matplot::transform(y, [](auto y)
                                      { return -y; }),
                "--xr");
  matplot::plot(x, matplot::transform(x, [](auto x)
                                      { return x / matplot::pi - 1.; }),
                "-:gs");
  matplot::plot({1.0, 0.7, 0.4, 0.0, -0.4, -0.7, -1}, "k");

  matplot::show();
  return 0;
}
int main(int argc, char *argv[])
{
  // test();
  // TestCase demo;
  // demo.mpcSolver();
  // demo.setupAndSolve();
  // gsmpl::KinematicTrajectoryOpti traj_opt;
  // traj_opt.solve();
  plot_test();
  return 0;
}
