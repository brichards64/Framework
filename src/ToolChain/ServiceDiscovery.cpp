#include "ServiceDiscovery.h"

ServiceDiscovery::ServiceDiscovery(std::string address, int multicastport, zmq::context_t * incontext){
  
  context=incontext;
  m_multicastport=multicastport;
  m_multicastaddress=address;

  
  pthread_create (&thread[0], NULL, ServiceDiscovery::MulticastListenThread, context);
  
  pthread_create (&thread[1], NULL, ServiceDiscovery::MulticastPublishThread, context);
  
  sleep(2);
  
  
  zmq::socket_t ServiceDiscovery (*context, ZMQ_REQ);
  //  std::stringstream socketSDstr;
  // socketSDstr<<"inproc://ServiceDiscovery"<<UUID;
  ServiceDiscovery.connect("inproc://ServiceDiscovery");
  
 
  zmq::socket_t ServicePublish (*context, ZMQ_PUSH);
  //std::stringstream socketSPstr;
  //socketSPstr<<"inproc://ServicePublish"<<UUID;
  
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



void* ServiceDiscovery::MulticastPublishThread(void* arg){
  
  //thread_args* args= static_cast<thread_args*>(arg);
  
  zmq::context_t * context = static_cast<zmq::context_t*>(arg);
  
  zmq::socket_t Ireceive (*context, ZMQ_PULL);
  // std::stringstream socketSPstr;
  //socketSPstr<<"inproc://ServicePublish"<<UUID;
  
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
  
  /* set up socket */
  sock = socket(AF_INET, SOCK_DGRAM, 0);
  //fcntl(sock, F_SETFL, O_NONBLOCK); 
  if (sock < 0) {
    perror("socket");
    exit(1);
  }
  bzero((char *)&addr, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  addrlen = sizeof(addr);
  
  /* send */
  addr.sin_addr.s_addr = inet_addr(group.c_str());
  
  
  
  //  Initialize poll set
  zmq::pollitem_t items [] = {
    { NULL, sock, ZMQ_POLLOUT, 0 },
    { Ireceive, 0, ZMQ_POLLIN, 0 }
  };
  
  
  bool running=true;
  
  while(running){
    
    zmq::poll (&items [0], 2, -1);
    
    if (items [0].revents & ZMQ_POLLOUT){
    
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
    
    }    
    
    if (items [1].revents & ZMQ_POLLIN) {

      zmq::message_t command;
      if(Ireceive.recv(&command)){
	std::stringstream tmp(static_cast<char*>(command.data()));
	if(tmp.str()=="Quit") running=false;
      }

    }

  }
  
  return (NULL);
  
  
}


void* ServiceDiscovery::MulticastListenThread(void* arg){
  
  // thread_args* args= static_cast<thread_args*>(arg);
  
  zmq::context_t * context = static_cast<zmq::context_t*>(arg);
  
  zmq::socket_t Ireceive (*context, ZMQ_REP);
  //std::stringstream socketSDstr;
  //socketSDstr<<"inproc://ServiceDiscovery"<<UUID;
  
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
  
  /* set up socket */
  sock = socket(AF_INET, SOCK_DGRAM, 0);
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &(int){ 1 }, sizeof(int));
  //fcntl(sock, F_SETFL, O_NONBLOCK); 
  if (sock < 0) {
    perror("socket");
    exit(1);
  }
  bzero((char *)&addr, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  addrlen = sizeof(addr);
  
  /* receive */
  if (bind(sock, (struct sockaddr *) &addr, sizeof(addr)) < 0) {        
    perror("bind");
    exit(1);
  }    
  mreq.imr_multiaddr.s_addr = inet_addr(group.c_str());         
  mreq.imr_interface.s_addr = htonl(INADDR_ANY);         
  if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP,&mreq, sizeof(mreq)) < 0) {
    perror("setsockopt mreq");
    exit(1);
  }         
  
  
  //////////////////////////////
  
  zmq::pollitem_t items [] = {
    { NULL, sock, ZMQ_POLLIN, 0 },
    { Ireceive, 0, ZMQ_POLLIN, 0 }
  };
  
  
  bool running=true;
  
  while(running){
    
    zmq::poll (&items [0], 1, -1);
    
    if (items [0].revents & ZMQ_POLLIN) {
      
      cnt = recvfrom(sock, message, sizeof(message), 0, (struct sockaddr *) &addr, (socklen_t*) &addrlen);
      if (cnt < 0) {
	//  perror("recvfrom");
	// exit(1);
      } 
      else if (cnt > 0) printf("%s: message = \"%s\"\n", inet_ntoa(addr.sin_addr), message);
      
      // do stuff her to keep track
    }
    
    if (items [1].revents & ZMQ_POLLIN) {
      
      zmq::message_t comm;
      
      if(Ireceive.recv(&comm)){
	
	std::istringstream iss(static_cast<char*>(comm.data()));
	
	zmq::message_t send(256);
	std::string tmp="return";
	snprintf ((char *) send.data(), 256 , "%s" ,tmp.c_str()) ;
	Ireceive.send(send);
	
      }  
      
    }
  }
  
  return (NULL);
  
}
