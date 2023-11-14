// A simple program that computes the square root of a number
#include <iostream>
#include <string>
#include "osqp_demo/osqp_demo.h"

int main(int argc, char* argv[])
{
  TestCase demo;
  demo.mpcSolver();
  // demo.setupAndSolve();
  return 0;
}
