// A simple program that computes the square root of a number
#include <iostream>
#include <string>
#include "traj_opt/src/test/osqp_demo.h"

int main(int argc, char *argv[])
{
  test();
  TestCase demo;
  demo.mpcSolver();
  demo.setupAndSolve();

  return 0;
}
