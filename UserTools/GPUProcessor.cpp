#include "GPUProcessor.h"

GPUProcessor::GPUProcessor():Tool(){}


bool GPUProcessor::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;


  m_data->triggeroutput=false;


  return true;
}


bool GPUProcessor::Execute(){

  //do stuff with m_data->Samples

  m_data->triggeroutput=true;

  return true;
}


bool GPUProcessor::Finalise(){

  return true;
}
