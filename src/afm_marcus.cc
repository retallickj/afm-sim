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
#include <stdio.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>

#if !(defined(_WIN32) || defined(_WIN64))
#include <Python.h>
#endif

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

  #if (defined(_WIN32) || defined(_WIN64))

  std::cout << "Calling Python as a system command..." << std::endl;
  std::string python_command;
  if (!problem.getParameter("python_interpreter").empty())
    python_command = problem.getParameter("python_interpreter");
  else if (!boost::filesystem::exists("C:\\Windows\\py.exe -3"))
    python_command = "C:\\Windows\\py.exe -3";
  else
    python_command = "python";

  // run the external program through the Windows shell
  std::string command = "\"" + python_command + " \"" + scriptPath() + "\" ";
  //std::string command = "\"py -3 \"" + scriptPath() + "\" ";
  command += "-i \"" + script_problem_path + "\" "; // problem path for the script to read
  command += "-o \"" + script_result_path + "\"\"";  // result path that the script writes to
  std::cout << "Calling command: " << command << std::endl;
  system(command.c_str());

  #else

  std::cout << "Calling Python with Python API..." << std::endl;
  Py_Initialize();
  PyObject *obj = Py_BuildValue("s", scriptPath().c_str());
  FILE* script_file = _Py_fopen_obj(obj, "r+");
  int argc = 5;
  wchar_t * argv[5];

  argv[0] = Py_DecodeLocale(scriptPath().c_str(), NULL);
  argv[1] = Py_DecodeLocale("-i", NULL);
  argv[2] = Py_DecodeLocale(script_problem_path.c_str(), NULL);
  argv[3] = Py_DecodeLocale("-o", NULL);
  argv[4] = Py_DecodeLocale(script_result_path.c_str(), NULL);

  Py_SetProgramName(argv[0]);
  PySys_SetArgv(argc, argv);
  PyRun_SimpleFile(script_file, scriptPath().c_str());
  Py_Finalize();

  #endif

  // use writeResultXml in phys_engine to export the result
  writeResultsXml();

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

  // convert db locations to lattice unit
  for (std::pair<float,float> db_loc : db_locs) {
    std::tuple<int,int,int> twod_loc_lu = angstrom2LatticeUnit(db_loc.first, db_loc.second);
    db_locs_lu.push_back(twod_loc_lu);
    //std::cout << "DB in Lattice Unit: x=" << x_lu << ", y=" << y_lu << ", b=" << b << std::endl;
  }

  // convert afm nodes to lattice unit
  if (problem.afmPathCount() > 0) {
    int sim_afm_path_ind = problem.simulateAFMPathInd();
    if (sim_afm_path_ind == -1) {
      std::cout << "No AFMPath index specified, this simulation will use the first path available." << std::endl;
      sim_afm_path_ind = 0;
    }
    std::shared_ptr<Problem::AFMPath> sim_afm_path = problem.getAFMPath(sim_afm_path_ind);
    for (std::shared_ptr<Problem::AFMNode> afmnode : sim_afm_path->nodes) {
      std::tuple<int,int,int> twod_loc_lu = angstrom2LatticeUnit(afmnode->x, afmnode->y);
      std::tuple<int,int,int,float> threed_loc_lu = std::tuple_cat(twod_loc_lu, std::make_tuple(afmnode->z));
      afm_node_locs.push_back(threed_loc_lu);
    }
  }

  // TODO implement path settings in GUI
  // TODO if there are multiple paths, there should be a way to determine which one to use for sim

  std::cout << "Writing minimized problem for AFMMarcus Python script..." << std::endl;
  std::cout << "Problem loc=" << script_problem_path << std::endl;

  // define major XML nodes
  bpt::ptree tree;
  bpt::ptree node_root;     // <min_problem>, root
  bpt::ptree node_params;   // <params>
  bpt::ptree node_dbs;      // <dbs>
  bpt::ptree node_afmnodes; // <afm_nodes>


  for (std::string param_key : problem.getParameterKeys()) {
    node_params.put(param_key, problem.getParameter(param_key));
  }

  // dbs
  for (std::tuple<int,int,int> dbl : db_locs_lu) {
    bpt::ptree node_dbdot;
    node_dbdot.put("<xmlattr>.x", std::to_string(std::get<0>(dbl)).c_str());
    node_dbdot.put("<xmlattr>.y", std::to_string(std::get<1>(dbl)).c_str());
    node_dbdot.put("<xmlattr>.b", std::to_string(std::get<2>(dbl)).c_str());
    node_dbs.add_child("dbdot", node_dbdot);
  }

  // afm nodes
  for (std::tuple<int,int,int,float> afm_node_loc : afm_node_locs) {
    bpt::ptree node_afmnode;
    node_afmnode.put("<xmlattr>.x", std::to_string(std::get<0>(afm_node_loc)).c_str());
    node_afmnode.put("<xmlattr>.y", std::to_string(std::get<1>(afm_node_loc)).c_str());
    node_afmnode.put("<xmlattr>.b", std::to_string(std::get<2>(afm_node_loc)).c_str());
    node_afmnode.put("<xmlattr>.z", std::to_string(std::get<3>(afm_node_loc)).c_str());
    node_afmnodes.add_child("afmnode", node_afmnode);
  }

  // add nodes to appropriate parent
  node_root.add_child("params", node_params);
  node_root.add_child("dbs", node_dbs);
  node_root.add_child("afmnodes", node_afmnodes);
  tree.add_child("min_problem", node_root);

  // write to file
  bpt::write_xml(script_problem_path, tree, std::locale(),
                    bpt::xml_writer_make_settings<std::string>(' ',4));

  std::cout << "Finished writing minimized problem" << std::endl;

  return true;
}

bool AFMMarcus::importResultsFromScript(const std::string &script_result_path)
{
  std::cout << std::endl << "***Importing results from script***" << std::endl;
  bpt::ptree tree; // boost property tree
  bpt::read_xml(script_result_path, tree, bpt::xml_parser::no_comments);

  // TODO catch read error exceptions

  // parse XML

  // read each afm_path
  for (bpt::ptree::value_type const &path_node : tree.get_child("afm_results")) {
    if (!path_node.first.compare("afm_path")) {
      line_scan_paths.push_back(readLineScanPath(path_node));
    }
  }

  std::cout << line_scan_paths.size() << " line scans recorded" << std::endl;

  return true;
}

phys::PhysicsEngine::LineScanPath AFMMarcus::readLineScanPath(bpt::ptree::value_type const &path_node)
{
  LineScanPath line_scan;

  line_scan.path_id = path_node.second.get<int>("<xmlattr>.path_id");

  // read db encoutered by this AFM path
  for (bpt::ptree::value_type const &db_node : path_node.second.get_child("dbs_encountered")) {
    std::pair<float,float> db_physloc = latticeUnit2Angstrom(
          db_node.second.get<int>("<xmlattr>.x"),
          db_node.second.get<int>("<xmlattr>.y"),
          db_node.second.get<int>("<xmlattr>.b")
        );
    line_scan.db_locs_enc.push_back(db_physloc);
    std::cout << "read db x=" << line_scan.db_locs_enc.back().first <<
        ", y=" << line_scan.db_locs_enc.back().second << std::endl;
  }

  // read line scan results
  for (bpt::ptree::value_type const &line_scan_node :
          path_node.second.get_child("scan_results")) {
    line_scan.results.push_back(line_scan_node.second.data());
    std::cout << "line scan result " << line_scan.results.back() << std::endl;
  }

  return line_scan;
}


std::tuple<int,int,int> AFMMarcus::angstrom2LatticeUnit(float x, float y)
{
  int x_lu = round(x / 3.84);
  int y_lu = round(y / 7.68);
  int b = round((y - 7.68*y_lu) / 2.4);
  //int b = round(fmod(y, 7.68) / 2.4);
  std::cout << "ang2lu (" << x << "," << y << ") to (" << x_lu << "," << y_lu <<
      "," << b << ")" << std::endl;
  return std::make_tuple(x_lu, y_lu, b);
}

std::pair<float,float> AFMMarcus::latticeUnit2Angstrom(int x_lu, int y_lu, int b)
{
  float x = static_cast<float>(x_lu) * 3.84;
  float y = static_cast<float>(y_lu) * 7.68 + static_cast<float>(b) * 2.4;
  std::cout << "lu2ang (" << x_lu << "," << y_lu << "," << b << ") to (" << x << "," << y << ")" << std::endl;
  return std::make_pair(x, y);
}
