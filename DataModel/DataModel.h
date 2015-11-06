#ifndef DATAMODEL_H
#define DATAMODEL_H

#include <map>
#include <string>
#include <vector>

//#include "TTree.h"

#include "Store.h"


class DataModel{


 public:
  
  DataModel();
  //TTree* GetTTree(std::string name);
  //void AddTTree(std::string name,TTree *tree);

  Store vars;


 private:
  
  //std::map<std::string,TTree*> m_trees; 
  
  
  
};



#endif
