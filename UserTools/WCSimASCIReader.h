#ifndef WCSimASCIReader_H
#define WCSimASCIReader_H

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Tool.h"

class WCSimASCIReader: public Tool {


 public:

  WCSimASCIReader();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:





};


#endif
