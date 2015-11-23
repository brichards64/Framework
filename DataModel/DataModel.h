#ifndef DATAMODEL_H
#define DATAMODEL_H

#include <map>
#include <string>
#include <vector>

//#include "TTree.h"

#include "Store.h"

class SubSample{

 public:
  
  SubSample();
  SubSample(int PMTid, int time){
    m_PMTid=PMTid;
    m_time=time;
  }
  
  int m_PMTid;
  int m_time;
  
};




class DataModel{


 public:
  
  DataModel();
  //TTree* GetTTree(std::string name);
  //void AddTTree(std::string name,TTree *tree);

  Store vars;
  std::vector<SubSample> Samples;
  bool triggeroutput;

 private:
  
  //std::map<std::string,TTree*> m_trees; 
  
  
  
};



#endif
