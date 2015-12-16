#ifndef TOOLCHAIN_H
#define TOOLCHAIN_H

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <pthread.h>
#include <time.h> 

#include "Tool.h"
#include "DataModel.h"
//#include "SocketCom.h"
#include "zmq.hpp"
#include "Factory.h"


class ToolChain{
  
 public:
  ToolChain(std::string configfile);
  ToolChain(bool verbose=true, int errorlevel=0); 
  //verbosity: true= print out status messages , false= print only error messages;
  //errorlevels: 0= do not exit; error 1= exit if unhandeled error ; exit 2= exit on handeled and unhandeled errors; 
  
  void Add(std::string name,Tool *tool,std::string configfile="");
  int Initialise();
  int Execute(int repeates=1);
  int Finalise();

  void Interactive();
  void Remote(int portnum, std::string SD_address="239.192.1.1", int SD_port=5000);
   
private:

  void Init();
  //void MulticastServiceDiscovery(std::string address, int multicastport);


  std::string ExecuteCommand(std::string connand);
  static  void *InteractiveThread(void* arg);
  //static  void *MulticastPublishThread(void* arg);
  //static  void *MulticastListenThread(void* arg);

  bool m_verbose;
  int m_errorlevel;
  std::vector<Tool*> m_tools;
  std::vector<std::string> m_toolnames;
  std::vector<std::string> m_configfiles;
  DataModel m_data;
  
  pthread_t thread[2];
  zmq::context_t *context;
  
  bool exeloop;
  long execounter;
  bool Initialised;
  bool Finalised;
  bool paused;
  
  int m_remoteport;
  int m_multicastport;
  std::string m_multicastaddress;
  long int UUID;
  
};

#endif
