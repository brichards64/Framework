#include "TriggerOutput.h"

TriggerOutput::TriggerOutput():Tool(){}


bool TriggerOutput::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;

  return true;
}


bool TriggerOutput::Execute(){

  std::string outfile;
  m_variables.Get("outfile",outfile);
  
  std::ofstream out (outfile.c_str());
  if (out.is_open())
    {
      out<<m_data->triggeroutput;
      out.close();
    }
  else {
    std::cout << "Unable to open file";
    return false;  
  }
  
  return true;
}


bool TriggerOutput::Finalise(){

  return true;
}
