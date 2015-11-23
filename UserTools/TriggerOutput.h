#ifndef TriggerOutput_H
#define TriggerOutput_H

#include <string>
#include <iostream>
#include <fstream>

#include "Tool.h"

class TriggerOutput: public Tool {


 public:

  TriggerOutput();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:





};


#endif
