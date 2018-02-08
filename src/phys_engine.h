// @file:     phys_engine.h
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2017.08.23 - Samuel
// @license:  GNU LGPL v3
//
// @desc:     Base class for physics engines

#include "problem.h"

#include <string>
#include <vector>
#include <boost/circular_buffer.hpp>

namespace phys{

  namespace bpt = boost::property_tree;

  class PhysicsEngine
  {
  public:

    // constructor
    PhysicsEngine(const std::string &eng_nm, const std::string &i_path, const std::string &o_path, const std::string &script_path="", const std::string &temp_path="");

    // destructor
    ~PhysicsEngine() {};

    // export results
    void writeResultsXml();

    // ACCESSORS
    std::string name() {return eng_name;}
    std::string scriptPath() {return script_path;}
    std::string inPath() {return in_path;}
    std::string outPath() {return out_path;}
    std::string tempPath() {return temp_path;}

    // variables
    Problem problem;

    std::vector<std::pair<float,float>> db_locs; // location of free dbs
    boost::circular_buffer<std::vector<int>> db_charges;

  private:
    std::string eng_name;     // name of this physics engine
    std::string script_path;  // for further invoking Python scripts
    std::string in_path;      // input file path containing the problem to run
    std::string out_path;     // output file path containing the simulation result
    std::string temp_path;    // path where temporary files will be generated

  };

}
