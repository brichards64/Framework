#include "ToolChain.h"

ToolChain::ToolChain(bool verbose,int errorlevel){

  context=new zmq::context_t(1);
  m_verbose=verbose;
  m_errorlevel=errorlevel;
  
  if(m_verbose){  
    std::cout<<"********************************************************"<<std::endl;
    std::cout<<"**** Tool chain created ****"<<std::endl;
    std::cout<<"********************************************************"<<std::endl<<std::endl;
  }

  execounter=0;
  Inialised=false;
  Finalised=false;

}



void ToolChain::Add(std::string name,Tool *tool,std::string configfile){
  
  if(m_verbose)std::cout<<"Adding Tool=\""<<name<<"\" tool chain"<<std::endl;
  m_tools.push_back(tool);
  m_toolnames.push_back(name);
  m_configfiles.push_back(configfile);
  if(m_verbose)std::cout<<"Tool=\""<<name<<"\" added successfully"<<std::endl<<std::endl; 
  
}



void ToolChain::Initialise(){
  
  if(m_verbose){
    std::cout<<"********************************************************"<<std::endl;
    std::cout<<"**** Initialising tools in toolchain ****"<<std::endl;
    std::cout<<"********************************************************"<<std::endl<<std::endl;
  }
  
  for(int i=0 ; i<m_tools.size();i++){  
    
    if(m_verbose) std::cout<<"Initialising "<<m_toolnames.at(i)<<std::endl;
    
    try{    
      if(m_tools.at(i)->Initialise(m_configfiles.at(i), m_data)){
	if(m_verbose)std::cout<<m_toolnames.at(i)<<" initialised successfully"<<std::endl<<std::endl;
      }
      else{
	std::cout<<"WARNING !!!!! "<<m_toolnames.at(i)<<" Failed to initialise (exit error code)"<<std::endl<<std::endl;
	if(m_errorlevel>1) exit(1);
      }
      
    }
    
    catch(...){
      std::cout<<"WARNING !!!!! "<<m_toolnames.at(i)<<" Failed to initialise (uncaught error)"<<std::endl<<std::endl;
      if(m_errorlevel>0) exit(1);
    }
    
  }
  
  if(m_verbose){std::cout<<"**** Tool chain initilised ****"<<std::endl;
    std::cout<<"********************************************************"<<std::endl<<std::endl;
  }

  execounter=0;
  Inialised=true;
  Finalised=false;

}



void ToolChain::Execute(int repeates){
  
  for(int i=0;i<repeates;i++){

    if(m_verbose){
      std::cout<<"********************************************************"<<std::endl;
      std::cout<<"**** Executing tools in toolchain ****"<<std::endl;
      std::cout<<"********************************************************"<<std::endl<<std::endl;
    }
    
    for(int i=0 ; i<m_tools.size();i++){
      
      if(m_verbose)    std::cout<<"Executing "<<m_toolnames.at(i)<<std::endl;
      
      try{
	if(m_tools.at(i)->Execute()){
	  if(m_verbose)std::cout<<m_toolnames.at(i)<<" executed  successfully"<<std::endl<<std::endl;
	}
	else{
	std::cout<<"WARNING !!!!!! "<<m_toolnames.at(i)<<" Failed to execute (error code)"<<std::endl<<std::endl;
	if(m_errorlevel>1)exit(1);
	}  
      }
      
      catch(...){
      std::cout<<"WARNING !!!!!! "<<m_toolnames.at(i)<<" Failed to execute (uncaught error)"<<std::endl<<std::endl;
      if(m_errorlevel>0)exit(1);
      }
      
    } 
    if(m_verbose){
      std::cout<<"**** Tool chain executed ****"<<std::endl;
      std::cout<<"********************************************************"<<std::endl<<std::endl;
    }
  }

  execounter++;

}



void ToolChain::Finalise(){
  if(m_verbose){
    std::cout<<"********************************************************"<<std::endl;
    std::cout<<"**** Finalising tools in toolchain ****"<<std::endl;
    std::cout<<"********************************************************"<<std::endl<<std::endl;
  }  
  
  for(int i=0 ; i<m_tools.size();i++){
    
    if(m_verbose)std::cout<<"Finalising "<<m_toolnames.at(i)<<std::endl;
    

    try{
      if(m_tools.at(i)->Finalise()){
	if(m_verbose)std::cout<<m_toolnames.at(i)<<" Finalised successfully"<<std::endl<<std::endl;
      }
      else{
	std::cout<<"WRNING !!!!!!! "<<m_toolnames.at(i)<<" Finalised successfully (error code)"<<std::endl<<std::endl;;
	if(m_errorlevel>1)exit(1);
      }  
    }
    
    catch(...){
      std::cout<<"WRNING !!!!!!! "<<m_toolnames.at(i)<<" Finalised successfully (uncaught error)"<<std::endl<<std::endl;;
      if(m_errorlevel>0)exit(1);
    }
    
  }
  
  if(m_verbose){
    std::cout<<"**** Tool chain Finalised ****"<<std::endl;
    std::cout<<"********************************************************"<<std::endl<<std::endl;
  }

  execounter=0;
  Inialised=false;
  Finalised=true;
}


void ToolChain::Interactive(){
  m_verbose=false;  
  exeloop=false;
  
  zmq::socket_t Ireceiver (*context, ZMQ_PAIR);
  Ireceiver.bind("inproc://control");
  
  pthread_create (&thread, NULL, ToolChain::InteractiveThread, context);
  
  while (true){

    zmq::message_t message;
    std::string command="";
    if(Ireceiver.recv (&message, ZMQ_NOBLOCK)){
      
      std::istringstream iss(static_cast<char*>(message.data()));
      iss >> command;

      std::cout<<"Please type command : Start, Pause, Unpause, Stop, Quit (Initialise, Execute, Finalise)"<<std::endl;
      std::cout<<">";
      
    }
    
    ExecuteCommand(command);
  }  
  
  
}  



void ToolChain::ExecuteCommand(std::string command){
  
  if(command=="Initialise") Initialise();
  else if (command=="Execute") Execute();
  else if (command=="Finalise") Finalise();
  else if (command=="Quit")exit(0);
  else if (command=="Start"){
    Initialise();
    exeloop=true;
  }
  else if(command=="Pause") exeloop=false;
  else if(command=="Unpause") exeloop=true;
  else if(command=="Stop") {
    exeloop=false;
    Finalise();
  }
  else if(command!="")std::cout<<"command not recognised please try again"<<std::endl;
 
  if(exeloop) Execute();
  
}




void ToolChain::Remote(int portnum){
  
  m_verbose=false;
  exeloop=false;

  std::stringstream tcp;
  tcp<<"tcp://*:"<<portnum;

  zmq::socket_t Ireceiver (*context, ZMQ_REP);
  Ireceiver.bind(tcp.str().c_str());
  
  while (true){
    zmq::message_t message;
    std::string command="";
    if(Ireceiver.recv(&message, ZMQ_NOBLOCK)){
      
      std::istringstream iss(static_cast<char*>(message.data()));
      iss >> command;

      zmq::message_t send(256);
      std::string tmp="got your message";
      snprintf ((char *) send.data(), 256 , "%s" ,tmp.c_str()) ;
      Ireceiver.send(send);
      
    }
    
    ExecuteCommand(command);
  }  
  
  
}



void* ToolChain::InteractiveThread(void* arg){

  zmq::context_t * context = static_cast<zmq::context_t*>(arg);

  zmq::socket_t Isend (*context, ZMQ_PAIR);
  Isend.connect("inproc://control");

  bool running=true;

  std::cout<<"Please type command : Start, Pause, Unpause, Stop, Quit (Initialise, Execute, Finalise)"<<std::endl;
  std::cout<<">";

  
  while (running){

    std::string tmp;
    std::cin>>tmp;
    zmq::message_t message(256);
    snprintf ((char *) message.data(), 256 , "%s" ,tmp.c_str()) ;
    Isend.send(message);

    if (tmp=="Quit")running=false;
  }

  return (NULL);

}

