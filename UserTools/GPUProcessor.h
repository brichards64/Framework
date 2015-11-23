#ifndef GPUProcessor_H
#define GPUProcessor_H

#include <string>
#include <iostream>

#include "Tool.h"

class GPUProcessor: public Tool {


 public:

  GPUProcessor();
  bool Initialise(std::string configfile,DataModel &data);
  bool Execute();
  bool Finalise();


 private:





};


#endif
