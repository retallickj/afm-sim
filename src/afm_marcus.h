// @file:     phys_engine.h
// @author:   Samuel
// @created:  2018.02.06
// @editted:  2017.02.06 - Samuel
// @license:  GNU LGPL v3
//
// @desc:     C++ wrapper that interfaces with db-sim and calls the Python scripts
//            for the AFMMarcus simulator

#include "phys_engine.h"

namespace phys {

  class AFMMarcus : public PhysicsEngine
  {
  public:
    AFMMarcus(const std::string &in_path, const std::string &out_path, const std::string &script_path, const std::string &temp_path);
    ~AFMMarcus() {};

    // run simulation, boolean value reports whether the simulation was successful
    // or not.
    bool runSim();

  private:

    // Export the problem to a file with only the content required by the Marcus script,
    // returns true if the write was successful
    bool exportProblemForScript(const std::string &script_problem_path);

    // Import the simulation result from the script, returns true if the read was successful
    bool importResultsFromScript(const std::string &script_result_path);
    phys::PhysicsEngine::LineScan readLineScan(bpt::ptree::value_type const &path_node);
    

    // VARS
    std::vector<std::pair<int,int>> db_locs_lu;   // DB locations in lattice unit


  }; // end of AFMMarcus class

} // end of phys namespace
