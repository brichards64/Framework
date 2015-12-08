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
  if (data.is_open()){
    std::vector<int>PMTid;
    std::vector<int>time;
    while ( getline (data,line) )
      {
	int tmpPMTid=0;
	int tmptime=0;
	std::stringstream tmp(line);
	tmp >> tmpPMTid>>tmptime;
	PMTid.push_back(tmpPMTid);
	time.push_back(tmptime);
	
      }

    SubSample tmpsb(PMTid,time);
    for(int i=0;i<100;i++)m_data->Samples.push_back(tmpsb);
    
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
