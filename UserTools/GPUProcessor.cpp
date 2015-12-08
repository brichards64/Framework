#include "GPUProcessor.h"
#include "../CUDA/library_daq.h"

GPUProcessor::GPUProcessor():Tool(){}


bool GPUProcessor::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;


  m_data->triggeroutput=false;


  gpu_daq_initialize();

  return true;
}


bool GPUProcessor::Execute(){

  //do stuff with m_data->Samples

  std::vector<int> PMTids;
  std::vector<int> times;

  for( std::vector<SubSample>::const_iterator is=m_data->Samples.begin(); is!=m_data->Samples.end(); ++is){
    PMTids.push_back(is->m_PMTid);
    times.push_back(is->m_time);
  }

  int the_output;
  the_output = CUDAFunction(PMTids, times);

  m_data->triggeroutput=(bool)the_output;

  return true;
}


bool GPUProcessor::Finalise(){

  gpu_daq_finalize();

  return true;
}
