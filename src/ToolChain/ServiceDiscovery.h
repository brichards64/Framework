#ifndef SERVICEDISCOVERY_H
#define SERVICEDISCOVERY_H

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
//#include <fcntl.h>
#include <string>

#include "zmq.hpp"

/*
struct thread_args{
  thread_args(long inUUID, zmq::context_t *incontext){
    UUID=inUUID;
    context=incontext;
  }
  long UUID;
  zmq::context_t *context;
};
*/

class ServiceDiscovery{
  
 public:
  
  ServiceDiscovery(std::string address, int multicastport, zmq::context_t * incontext);
  
  
  
 private:
  
  long UUID;
  zmq::context_t *context;
  pthread_t thread[2];
  int m_multicastport;
  std::string m_multicastaddress;
  static  void *MulticastPublishThread(void* arg);
  static  void *MulticastListenThread(void* arg);
  
  
};

#endif
