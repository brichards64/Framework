#ifndef SOCKETCOM_H
#define SOCKETCOM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <iostream>
#include <netdb.h> 
#include <pthread.h>
#include <fcntl.h>

class SocketCom{
  
 public:
  
  SocketCom(bool client, std::string hostname, int port, char *buffer,bool *newbuff); //char buffer[256]
  void ListenStart();
  void ListenStop();
static  void *Listen(void* arg);
static  void *ListenThread(void* arg);
 bool Connect();
 bool Disconnect();
 bool Send();
 

  struct sockaddr_in serv_addr, srcaddr,cli_addr[5];
  int sockfd, newsockfd[5], portno, n;
  socklen_t clilen[5];
  char* m_buffer;
  bool* m_newbuff;
 
  struct hostent *server;


  pthread_t threads[5];
  int tcontrol[5];
  pthread_mutex_t mu_tcontrol;
  pthread_mutex_t mu_newbuff;

 private:

  /*
    struct sockaddr_in
    {
    short   sin_family; // must be AF_INET 
    u_short sin_port;
    struct  in_addr sin_addr;
    char    sin_zero[8]; // Not used, must be zero //
    };
  */
  
  
};

struct input{

public:

  int* tnum ;
  SocketCom* self;

};

#endif
