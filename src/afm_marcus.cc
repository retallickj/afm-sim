// @file:     phys_engine.cc
// @author:   Samuel
// @created:  2018.02.06
// @editted:  2017.02.06 - Samuel
// @license:  GNU LGPL v3
//
// @desc:     C++ wrapper that interfaces with db-sim and calls the Python scripts
//            for the AFMMarcus simulator

#include "afm_marcus.h"

#include <math.h>
#include <boost/filesystem.hpp>

using namespace phys;

// constructor
AFMMarcus::AFMMarcus(const std::string &in_path, const std::string &out_path, const std::string &script_path, const std::string &temp_path)
  : PhysicsEngine("AFMMarcus", in_path, out_path, script_path, temp_path)
{
}

// run the simulation
bool AFMMarcus::runSim()
{
  std::cout << std::endl << "*** AFMMarcus: Entered simulation ***" << std::endl;
  // check if script exists at script_path
  // TODO

  // determine the paths for exporting the minimized problem to the script, and 
  // importing the results from the script
  // FUTURE move all these temp stuff over to phys_engine
  std::string tmp_dir, script_problem_path, script_result_path;
  if (!tempPath().compare("")) {
    // use supplied temp path if available
    tmp_dir = tempPath();
  } else {
    // get current time formatted yyyymmdd-hhmmss
    std::string curr_time_str = formattedTime("%Y%m%d-%H%M%S");

    tmp_dir = boost::filesystem::temp_directory_path().string();
    tmp_dir += "/afm_marcus/" + curr_time_str;
    std::cout << "Temp directory = " << tmp_dir << std::endl;
  }
  boost::filesystem::create_directories(tmp_dir);
  script_problem_path = tmp_dir + "/afmmarcus_problem.xml";
  script_result_path = tmp_dir + "/afmmarcus_result.xml";

  // write the minimized problem to file with only the info required for python script
  // to understand the problem
  if (!exportProblemForScript(script_problem_path))
    return false;

  // setup command for invoking python script with the input and output file paths as 
  // arguments. NOTE might not work on Windows, look into PyRun_SimpleFile if necessary.
  std::string command = "python " + scriptPath() + " ";
  command += script_problem_path + " "; // problem path for the script to read
  command += script_result_path;        // result path that the script writes to

  // call the script. This is non-forking so current process will wait until it finishes
  system(command.c_str());

  // parse the outputs from the result file generated by the script
  if (!importResultFromScript(script_result_path))
    return false;

  // use writeResultXml in phys_engine to export the result, TODO with flags indicating
  // what is being written
  writeResultsXml();
  // TODO update phys_engine's writeResultXml to be able to output the results
  //      from this simulator

  std::cout << std::endl << "*** AFMMarcus: Simulation has completed ***" << std::endl;
  return true;
}


// PRIVATE

bool AFMMarcus::exportProblemForScript(const std::string &script_problem_path)
{
  // for AFMMarcus, only the following items are needed:
  //   AFM Path
  //   DB location in lattice units
  //   Other simulation parameters
  // NOTE y lattice units are WRONG right now. Don't worry at this point but deal with it when 2D scans are considered. TODO

  // convert db locations to lattice unit
  for (std::pair<float,float> db_loc : db_locs) {
    int x_lu = round(db_loc.first / 3.84);
    int y_lu = round(db_loc.second / 7.68) + round(fmod(db_loc.second, 7.68) / 2.4); // TODO wrong, fix later
    db_locs_lu.push_back(std::make_pair(x_lu, y_lu));
    std::cout << "DB in Lattice Unit: x=" << x_lu << ", y=" << y_lu << std::endl;
  }

  // convert afm nodes to lattice unit
  std::vector<std::tuple<int,int,float>> afm_nodes_loc;
  std::shared_ptr<Problem::AFMPath> sim_afm_path = problem.getAFMPath(problem.simulateAFMPathInd());
  for (std::shared_ptr<Problem::AFMNode> afmnode : sim_afm_path->nodes) {
    int x_lu = round(afmnode->x / 3.84);
    int y_lu = round(afmnode->y / 7.68) + round(fmod(afmnode->y, 7.68) / 2.4); // TODO also wrong
    float z = afmnode->z;
    afm_nodes_loc.push_back(std::make_tuple(x_lu, y_lu, z));
  }

  // TODO implement path settings in GUI
  // TODO if there are multiple paths, there should be a way to determine which one to use for sim

  std::cout << "Writing minimized problem for AFMMarcus Python script..." << std::endl;
  std::cout << "Problem loc=" << script_problem_path << std::endl;

  // define major XML nodes
  bpt::ptree tree;
  bpt::ptree node_root;     // <min_problem>
  bpt::ptree node_dbs;      // <dbs>
  bpt::ptree node_afmnodes; // <afm_nodes>

  // dbs
  for (std::pair<int,int> dbl : db_locs_lu) {
    bpt::ptree node_dbdot;
    node_dbdot.put("<xmlattr>.x", std::to_string(dbl.first).c_str());
    node_dbdot.put("<xmlattr>.y", std::to_string(dbl.second).c_str());
    node_dbs.add_child("dbdot", node_dbdot);
  }

  // afm nodes
  /* adapt this in the future when 2D scans are considered
  for (std::shared_ptr<Problem::AFMNode> afmnode : sim_afm_path->nodes) {
    bpt::ptree node_afmnode;
    node_afmnode.put("<xmlattr>.x", std::to_string(afmnode->x).c_str());
    node_afmnode.put("<xmlattr>.y", std::to_string(afmnode->y).c_str());
    node_afmnode.put("<xmlattr>.z", std::to_string(afmnode->z).c_str());
    node_afmnodes.add_child("afmnode", node_afmnode);
  }*/
  for (std::tuple<int,int,float> afm_node_loc : afm_nodes_loc) {
    bpt::ptree node_afmnode;
    node_afmnode.put("<xmlattr>.x", std::to_string(std::get<0>(afm_node_loc)).c_str());
    node_afmnode.put("<xmlattr>.y", std::to_string(std::get<1>(afm_node_loc)).c_str());
    node_afmnode.put("<xmlattr>.z", std::to_string(std::get<2>(afm_node_loc)).c_str());
    node_afmnodes.add_child("afmnode", node_afmnode);
  }

  // add nodes to appropriate parent
  node_root.add_child("dbs", node_dbs);
  node_root.add_child("afmnodes", node_afmnodes);
  tree.add_child("min_problem", node_root);

  // write to file
  bpt::write_xml(script_problem_path, tree, std::locale(), 
                    bpt::xml_writer_make_settings<std::string>(' ',4));

  std::cout << "Finished writing minimized problem" << std::endl;

  return true;
}

bool AFMMarcus::importResultFromScript(const std::string &script_result_path)
{

  return false;
}
