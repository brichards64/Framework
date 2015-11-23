#include "WCSimASCIReader.h"

WCSimASCIReader::WCSimASCIReader():Tool(){}


bool WCSimASCIReader::Initialise(std::string configfile, DataModel &data){

  if(configfile!="")  m_variables.Initialise(configfile);
  //m_variables.Print();

  m_data= &data;

  return true;
}


bool WCSimASCIReader::Execute(){
  
  std::string inputfile;
  m_variables.Get("inputfile",inputfile);
  std::string line;
  std::ifstream data (inputfile.c_str());
  if (data.is_open())
    {
      while ( getline (data,line) )
	{
	  int PMTid=0;
	  int time=0;
	  std::stringstream tmp(line);
	  tmp >> PMTid>>time;

	  SubSample tmpsb(PMTid,time);
	  m_data->Samples.push_back(tmpsb);
	}
      data.close();
    }
  
  else {
    std::cout << "Unable to open file"; 
    return false;
  }  
  
  return true;
}


bool WCSimASCIReader::Finalise(){
  
  return true;
}
