#include "ToolChain.h"
#include "ServiceDiscovery.cpp"

ToolChain::ToolChain(std::string configfile){
 
  Store config;
  config.Initialise(configfile);
  config.Get("verbose",m_verbose);
  config.Get("error_level",m_errorlevel);
  config.Get("remote_port",m_remoteport);
  config.Get("service_discovery_address",m_multicastaddress);
  config.Get("service_discovery_port",m_multicastport);
 
  Init();

  std::string toolsfile="";
  config.Get("Tools_File",toolsfile);
  
  if(toolsfile!=""){
    std::ifstream file(toolsfile.c_str());
    std::string line;
    if(file.is_open()){
      
      while (getline(file,line)){
	if (line.size()>0){
	  if (line.at(0)=='#')continue;
	  std::string name;
	  std::string tool;
	  std::string conf;
	  std::stringstream stream(line);
	  
	  if(stream>>name>>tool>>conf) Add(name,Factory(tool),conf);
	  
	}
	
      }
    }
    
    file.close();
    
  }

  int Inline=0;
  bool interactive=false;
  bool remote=false;
  config.Get("Inline",Inline);
  config.Get("Interactive",interactive);
  config.Get("Remote",remote);
  if(Inline>0){
    Initialise();
    Execute(Inline);
    Finalise();
  }
  else if(interactive)Interactive();
  else if(remote)Remote(m_remoteport, m_multicastaddress, m_multicastport);
  
}

ToolChain::ToolChain(bool verbose,int errorlevel){

  m_verbose=verbose;
  m_errorlevel=errorlevel;
  Init();
  
}

void ToolChain::Init(){

 srand (time(NULL));
 UUID=rand() % 1000000000000 + 1;

 std::cout<<"UUID = "<<UUID<<std::endl;

  context=new zmq::context_t(1);
  //add context to data model;
  
  if(m_verbose){  
    std::cout<<"********************************************************"<<std::endl;
    std::cout<<"**** Tool chain created ****"<<std::endl;
    std::cout<<"********************************************************"<<std::endl<<std::endl;
    }
  
  execounter=0;
  Initialised=false;
  Finalised=true;
  paused=false;
}



void ToolChain::Add(std::string name,Tool *tool,std::string configfile){
  if(tool!=0){
    if(m_verbose)std::cout<<"Adding Tool=\""<<name<<"\" tool chain"<<std::endl;
    m_tools.push_back(tool);
    m_toolnames.push_back(name);
    m_configfiles.push_back(configfile);
    if(m_verbose)std::cout<<"Tool=\""<<name<<"\" added successfully"<<std::endl<<std::endl; 
  }
}



int ToolChain::Initialise(){

  bool result=0;

  if (Finalised){
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
	  result=1;
	  if(m_errorlevel>1) exit(1);
	}
	
      }
      
      catch(...){
	std::cout<<"WARNING !!!!! "<<m_toolnames.at(i)<<" Failed to initialise (uncaught error)"<<std::endl<<std::endl;
	result=2;
	if(m_errorlevel>0) exit(1);
      }
      
    }
    
    if(m_verbose){std::cout<<"**** Tool chain initilised ****"<<std::endl;
      std::cout<<"********************************************************"<<std::endl<<std::endl;
    }
    
    execounter=0;
    Initialised=true;
    Finalised=false;
  }
  else {
    std::cout<<"********************************************************"<<std::endl<<std::endl;
    std::cout<<" ERROR: ToolChain Cannot Be Initialised as already running. Finalise old chain first"<<std::endl;
    std::cout<<"********************************************************"<<std::endl<<std::endl;
    result=-1;
  }
  
  return result;
}



int ToolChain::Execute(int repeates){
 
  int result =0;
  
  if(Initialised){
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
	    result=1;
	    if(m_errorlevel>1)exit(1);
	  }  
	}
	
	catch(...){
	  std::cout<<"WARNING !!!!!! "<<m_toolnames.at(i)<<" Failed to execute (uncaught error)"<<std::endl<<std::endl;
	  result=2;
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
  
  else {
    std::cout<<"********************************************************"<<std::endl<<std::endl;
    std::cout<<" ERROR: ToolChain Cannot Be Executed As Has Not Been Initialised yet."<<std::endl;
    std::cout<<"********************************************************"<<std::endl<<std::endl;
    result=-1;
  }

  return result;
}



int ToolChain::Finalise(){
  
  int result=0;

  if(Initialised){
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
	  result=1;
	  if(m_errorlevel>1)exit(1);
	}  
      }
      
      catch(...){
	std::cout<<"WRNING !!!!!!! "<<m_toolnames.at(i)<<" Finalised successfully (uncaught error)"<<std::endl<<std::endl;
	result=2;
	if(m_errorlevel>0)exit(1);
      }
      
    }
    
  if(m_verbose){
    std::cout<<"**** Tool chain Finalised ****"<<std::endl;
    std::cout<<"********************************************************"<<std::endl<<std::endl;
  }
  
  execounter=0;
  Initialised=false;
  Finalised=true;
  paused=false;
  }
  
  else {
    std::cout<<"********************************************************"<<std::endl<<std::endl;
    std::cout<<" ERROR: ToolChain Cannot Be Finalised As Has Not Been Initialised yet."<<std::endl;
    std::cout<<"********************************************************"<<std::endl<<std::endl;
    result=-1;
  }
  
  return result;
}


void ToolChain::Interactive(){
  m_verbose=false;  
  exeloop=false;
  
  zmq::socket_t Ireceiver (*context, ZMQ_PAIR);
  Ireceiver.bind("inproc://control");
  
  pthread_create (&thread[0], NULL, ToolChain::InteractiveThread, context);
  
  while (true){
    
    zmq::message_t message;
    std::string command="";
    if(Ireceiver.recv (&message, ZMQ_NOBLOCK)){
      
      std::istringstream iss(static_cast<char*>(message.data()));
      iss >> command;
      
      std::cout<<ExecuteCommand(command)<<std::endl<<std::endl;
      command="";
      std::cout<<"Please type command : Start, Pause, Unpause, Stop, Quit (Initialise, Execute, Finalise)"<<std::endl;
      std::cout<<">";
      
    }
    
    ExecuteCommand(command);
  }  
  
  
}  



std::string ToolChain::ExecuteCommand(std::string command){
  std::string returnmsg="";
  
  if(command=="Initialise"){
    Initialise();
    returnmsg="Initialising ToolChain";
  }
  else if (command=="Execute"){
    Execute();
    returnmsg="Executing ToolChain";
  }
  else if (command=="Finalise"){
    Finalise();
    returnmsg="Finalising  ToolChain";
  }
  else if (command=="Quit")exit(0);
  else if (command=="Start"){
    Initialise();
    exeloop=true;
    returnmsg="Starting ToolChain";
  }
  else if(command=="Pause"){
    exeloop=false;
    paused=true;
    returnmsg="Pausing ToolChain";
  }
  else if(command=="Unpause"){
    exeloop=true;
    paused=false;
    returnmsg="Unpausing ToolChain";
  }
  else if(command=="Stop") {
    exeloop=false;
    Finalise();
    returnmsg="Stopping ToolChain";
  }
  else if(command=="Status"){
    std::stringstream tmp;
    if(Finalised) tmp<<"Waiting to Initialise ToolChain";
    if(Initialised && execounter==0) tmp<<"Initialised waiting to Execute ToolChain";
    if(Initialised && execounter>0){
      if(paused)tmp<<"ToolChain execution pasued";
      else tmp<<"ToolChain running (loop counter="<<execounter<<")";
    }
    returnmsg=tmp.str();
  }
  else if(command!=""){
    std::cout<<"command not recognised please try again"<<std::endl;
    returnmsg="command not recognised please try again";
  }

  if(exeloop) Execute();
  return returnmsg;
}




void ToolChain::Remote(int portnum, std::string SD_address, int SD_port){

  m_remoteport=portnum;
  m_multicastport=SD_port;
  m_multicastaddress=SD_address;
  m_verbose=false;
  exeloop=false;

  ServiceDiscovery *SD=new ServiceDiscovery(m_multicastaddress.c_str(),m_multicastport,context);

  
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
      std::string tmp=ExecuteCommand(command);
      if(tmp=="")tmp="got your message";
      snprintf ((char *) send.data(), 256 , "%s" ,tmp.c_str()) ;
      Ireceiver.send(send);
      
      std::cout<<"received "<<command<<std::endl;
      if(command=="Quit"){
	
	zmq::socket_t ServicePublisher (*context, ZMQ_PAIR);
	std::stringstream socketSPstr;
	socketSPstr<<"inproc://ServicePublish"<<UUID;
	ServicePublisher.connect(socketSPstr.str().c_str());
	zmq::socket_t ServiceDiscovery (*context, ZMQ_PAIR);	
	std::stringstream socketSDstr;
	socketSDstr<<"inproc://ServiceDiscovery"<<UUID;
	ServiceDiscovery.connect(socketSDstr.str().c_str());
	zmq::message_t Qsignal1(256);
	zmq::message_t Qsignal2(256);
	std::string tmp="Quit";
	snprintf ((char *) Qsignal1.data(), 256 , "%s" ,tmp.c_str()) ;
	snprintf ((char *) Qsignal2.data(), 256 , "%s" ,tmp.c_str()) ;
	ServicePublisher.send(Qsignal1);
	ServiceDiscovery.send(Qsignal2);
	
      }
      if(command!="")std::cout<<"sending command "<<command<<std::endl;    
      std::cout<<ExecuteCommand(command)<<std::endl<<std::endl;      
      command=""; 
      if(command!="")std::cout<<"done "<<command<<std::endl;   
    }
  
    ExecuteCommand(command);
   
  }  
  
  
}



void* ToolChain::InteractiveThread(void* arg){

  zmq::context_t * context = static_cast<zmq::context_t*>(arg);

  zmq::socket_t Isend (*context, ZMQ_PAIR);

  std::stringstream socketstr;
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

/*
void ToolChain::MulticastServiceDiscovery(std::string address,int multicastport){

  m_multicastport=multicastport;
  m_multicastaddress=address;
 
  pthread_create (&thread[0], NULL, ToolChain::MulticastListenThread, context);
  
  pthread_create (&thread[1], NULL, ToolChain::MulticastPublishThread, context);
  
  sleep(2);
  zmq::socket_t ServiceDiscovery (*context, ZMQ_REQ);
  ServiceDiscovery.connect("inproc://ServiceDiscovery");

  zmq::socket_t ServicePublish (*context, ZMQ_PUSH);
  ServicePublish.connect("inproc://ServicePublish");

 
  zmq::message_t send1(256);
  snprintf ((char *) send1.data(), 256 , "%s %d" ,m_multicastaddress.c_str(), m_multicastport);

zmq::message_t send2(256);
  snprintf ((char *) send2.data(), 256 , "%s %d" ,m_multicastaddress.c_str(), m_multicastport);
  
  zmq::message_t message;
  ServicePublish.send(send2); 
  ServiceDiscovery.send(send1);
  ServiceDiscovery.recv(&message); 


}

/*
void* ToolChain::MulticastPublishThread(void* arg){
  
  zmq::context_t * context = static_cast<zmq::context_t*>(arg);

  zmq::socket_t Ireceive (*context, ZMQ_PAIR);
  Ireceive.connect("inproc://multicastsetup");

  zmq::message_t message;
  Ireceive.recv (&message);
  std::istringstream remoteport(static_cast<char*>(message.data()));

  zmq::socket_t StatusReceiver (*context, ZMQ_REQ);
  StatusReceiver.connect(remoteport.str().c_str());

  Ireceive.recv (&message);
  std::istringstream multicast(static_cast<char*>(message.data()));

   zmq::socket_t MultiCastPub (*context, ZMQ_PUB);
    MultiCastPub.connect(multicast.str().c_str());

  bool running=true;

  while(running){


    zmq::message_t send(256);
    std::string tmp="Status";
    snprintf ((char *) send.data(), 256 , "%s" ,tmp.c_str()) ;
    StatusReceiver.send(send);
    
    zmq::message_t statusmessage;
    StatusReceiver.recv (&statusmessage);
    std::istringstream status(static_cast<char*>(statusmessage.data()));

    //  zmq::socket_t MultiCastPub (*context, ZMQ_PUB);
    // MultiCastPub.connect(multicast.str().c_str());
    
    zmq::message_t send2(256);
    std::string subject="Service";
    std::string from="myip";
    std::string process="myprocess";
    snprintf ((char *) send2.data(), 256 , "%s %s %s %s" , subject.c_str(), from.c_str(), process.c_str(),  status.str().c_str()) ;
    
    //zmq::socket_t MultiCastPub (*context, ZMQ_PUB);
    // MultiCastPub.connect(multicast.str().c_str());
    MultiCastPub.send(send2);
    // sleep(1);
    // MultiCastPub.disconnect(multicast.str().c_str());
    
    if(status.str()=="Quit")running=false;
    sleep(10);
    
    
  }
  
  return (NULL);

  
}
*/

/*
void* ToolChain::MulticastPublishThread(void* arg){

  zmq::context_t * context = static_cast<zmq::context_t*>(arg);
  
  zmq::socket_t Ireceive (*context, ZMQ_PULL);
  Ireceive.bind("inproc://ServicePublish");  

  zmq::message_t config;
  Ireceive.recv (&config);
  std::istringstream configuration(static_cast<char*>(config.data()));
  std::string group;
  int port;
  configuration>>group>>port;
  
  /// multi cast /////


   struct sockaddr_in addr;
   int addrlen, sock, cnt;
   struct ip_mreq mreq;
   char message[256];
   
   /* set up socket *//*
   sock = socket(AF_INET, SOCK_DGRAM, 0);
   fcntl(sock, F_SETFL, O_NONBLOCK); 
   if (sock < 0) {
     perror("socket");
     exit(1);
   }
   bzero((char *)&addr, sizeof(addr));
   addr.sin_family = AF_INET;
   addr.sin_addr.s_addr = htonl(INADDR_ANY);
   addr.sin_port = htons(port);
   addrlen = sizeof(addr);
   
   /* send *//*
   addr.sin_addr.s_addr = inet_addr(group.c_str());
   
   bool running=true;
   
   while(running){
     
     std::string subject="Discovery";
     std::string from="myip";
     std::string process="Service";
     std::string content="Online";
     snprintf (message, 256 , "%s %s %s %s" , subject.c_str(), from.c_str(), process.c_str(),  content.c_str()) ;
     printf("sending: %s\n", message);
     cnt = sendto(sock, message, sizeof(message), 0,(struct sockaddr *) &addr, addrlen);
     if (cnt < 0) {
       perror("sendto");
       exit(1);
     }
     sleep(5);
     
     zmq::message_t command;
     if(Ireceive.recv(&command, ZMQ_NOBLOCK)){
       std::stringstream tmp(static_cast<char*>(command.data()));
       if(tmp.str()=="Quit") running=false;
     }
     
   }
   
   return (NULL);
   
   
}

void* ToolChain::MulticastListenThread(void* arg){

  zmq::context_t * context = static_cast<zmq::context_t*>(arg);
  
  zmq::socket_t Ireceive (*context, ZMQ_REP);
  Ireceive.bind("inproc://ServiceDiscovery");  

  zmq::message_t config;
  Ireceive.recv (&config);
  std::istringstream configuration(static_cast<char*>(config.data()));
  std::string group;
  int port;
  configuration>>group>>port;
  
  Ireceive.send(config);

  ///// multi cast /////



   struct sockaddr_in addr;
   int addrlen, sock, cnt;
   struct ip_mreq mreq;
   char message[256];

   /* set up socket *//*
   sock = socket(AF_INET, SOCK_DGRAM, 0);
   fcntl(sock, F_SETFL, O_NONBLOCK); 
  if (sock < 0) {
     perror("socket");
     exit(1);
   }
   bzero((char *)&addr, sizeof(addr));
   addr.sin_family = AF_INET;
   addr.sin_addr.s_addr = htonl(INADDR_ANY);
   addr.sin_port = htons(port);
   addrlen = sizeof(addr);
   
   /* receive *//*
      if (bind(sock, (struct sockaddr *) &addr, sizeof(addr)) < 0) {        
         perror("bind");
	 exit(1);
      }    
      mreq.imr_multiaddr.s_addr = inet_addr(group.c_str());         
      mreq.imr_interface.s_addr = htonl(INADDR_ANY);         
      if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP,
		     &mreq, sizeof(mreq)) < 0) {
	 perror("setsockopt mreq");
	 exit(1);
      }         


      //////////////////////////////

  bool running=true;

  while(running){
  
  zmq::message_t command;
    if(Ireceive.recv(&command, ZMQ_NOBLOCK)){
      
      std::istringstream iss(static_cast<char*>(message.data()));
      iss >> command;
      
      zmq::message_t send(256);
      std::string tmp="return";
      snprintf ((char *) send.data(), 256 , "%s" ,tmp.c_str()) ;
      Ireceiver.send(send);
      
    }  
    
    cnt = recvfrom(sock, message, sizeof(message), 0, (struct sockaddr *) &addr, (socklen_t*) &addrlen);
    if (cnt < 0) {
      //  perror("recvfrom");
      // exit(1);
    } 
    else if (cnt > 0) printf("%s: message = \"%s\"\n", inet_ntoa(addr.sin_addr), message);
    
    // do stuff her to keep track
  }
  
  return (NULL);
  
  
}
*/
