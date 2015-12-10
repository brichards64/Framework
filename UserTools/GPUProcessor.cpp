#include "GPUProcessor.h"
#include "CUDA/library_daq.h"

GPUProcessor::GPUProcessor():Tool(){}


bool GPUProcessor::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;

  m_data->triggeroutput=false;



  
  std::string PMTFile;
  std::string DetectorFile;
  std::string ParameterFile;
  

  m_variables.Get("PMTFile",PMTFile);
  m_variables.Get("DetectorFile",DetectorFile);
  m_variables.Get("ParameterFile",ParameterFile);
  
  gpu_daq_initialize(PMTFile,DetectorFile,ParameterFile);

  // can acess variables directly like this and would be good if you could impliment in your code

  float dark_rate;
  m_variables.Get("dark_rate",dark_rate);

  //to do this in your code instead of passing the three strings you could just do

  //gpu_daq_initialize(m_variables);

  // then in your code you can include 
  //#include "Store.h"
  //  gpu_daq_initialize(Store variables);

  //and pull them out with the get function in the same way 

  return true;
}


bool GPUProcessor::Execute(){

  //do stuff with m_data->Samples
  /*
  for( std::vector<SubSample>::const_iterator is=m_data->Samples.begin(); is!=m_data->Samples.end(); ++is){
    PMTids.push_back(is->m_PMTid);
    times.push_back(is->m_time);
  }
  */
  int the_output;
  the_output = CUDAFunction(m_data->Samples.at(0).m_PMTid, m_data->Samples.at(0).m_time);

  m_data->triggeroutput=(bool)the_output;

  return true;
}


bool GPUProcessor::Finalise(){

  gpu_daq_finalize();

  return true;
}
