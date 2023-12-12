// A simple program that computes the square root of a number
#include <iostream>
#include <string>
#include <matplot/matplot.h>

#include "traj_opt/src/test/osqp_demo.h"
#include "traj_opt/src/trajectory/kinematic_trajectory_optimization.h"

int main(int argc, char *argv[])
{
  // test();
  // TestCase demo;
  // demo.mpcSolver();
  // demo.setupAndSolve();
  gsmpl::KinematicTrajectoryOpti traj_opt;
  traj_opt.plot_traj();
  return 0;
}
