#include "SocketCom.h"

SocketCom::SocketCom(bool client,std::string hostname, int port, char *buffer, bool *newbuff){
  
  m_buffer= buffer;
  m_newbuff=newbuff;
  
  sockfd=socket(AF_INET, SOCK_STREAM, 0);//create socket (adress domain, socket type, protacall 0=auto) 

  
  if (sockfd<0) std::cout<<"Failed to open socket"<<std::endl; 
  
  if(client){
    server = gethostbyname(hostname.c_str());
    if (server == NULL) {
      std::cout<<"ERROR, no such host"<<std::endl;
    }  
  }
  
  bzero((char *) &serv_addr, sizeof(serv_addr)); // has to be zeroed
  portno = port;
  serv_addr.sin_family = AF_INET; //interet based adressing
  
  if(!client) serv_addr.sin_addr.s_addr = INADDR_ANY; //own comps ip}
  serv_addr.sin_port = htons(portno); //prt /horns converts to network byte order
  
  if(client){
    memset(&srcaddr, 0, sizeof(srcaddr));
    srcaddr.sin_family = AF_INET;
    srcaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    srcaddr.sin_port = htons(portno+1);
    
    if(bind(sockfd, (struct sockaddr *) &srcaddr, sizeof(srcaddr))<0)std::cout<<"Failed to bind socket"<<std::endl;
  }
  
  if(!client){
    //  fcntl(sockfd, F_SETFL, O_NONBLOCK);
    if (bind(sockfd, (struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)   std::cout<<"Failed to bind socket"<<std::endl;
  }
}




void SocketCom::ListenStart(){
  pthread_mutex_init (&mu_tcontrol,NULL);
  pthread_mutex_init (&mu_newbuff,NULL);
  
  
  //   std::cout<<"in thread listen start "<<std::endl;
  for (int i=0; i<5;i++){
    tcontrol[i]=-1;
  }
  // std::cout<<"creating thread 0 "<<std::endl;
  pthread_create (&threads[0], NULL, SocketCom::Listen, this); 
  // std::cout<<"created thread 0 "<<std::endl;
  
}

void SocketCom::ListenStop(){
  // std::cout<<"in listen stop "<<std::endl;

  pthread_mutex_lock (&mu_tcontrol);
  for (int i=0; i<5;i++){
    tcontrol[i]=0;
    //pthread_cancel(threads[i]);
  }
  //std::cout<<"sending stop signals"<<std::endl;
  pthread_mutex_unlock (&mu_tcontrol);
  //std::cout<<"sent stop signals "<<std::endl;
  
}



void *SocketCom::Listen(void* arg){


  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS,NULL);
  
  SocketCom* self=(SocketCom*)arg;
  //std::cout<<"in thread 0 "<<std::endl;
  pthread_mutex_lock (&self->mu_tcontrol);
  self->tcontrol[0]=1;
  while(self->tcontrol[0]>0){
    pthread_mutex_unlock (&self->mu_tcontrol);
    //     std::cout<<"in while "<<std::endl;
    
    for(int tnum=1;tnum<5;){
      //    std::cout<<"tnum loop  "<<tnum<<std::endl;
      pthread_mutex_lock (&self->mu_tcontrol);      
      //     std::cout<<"test1  "<<std::endl;
      if (self->tcontrol[tnum]==-1){
	//          std::cout<<"test2  "<<std::endl;
	pthread_mutex_unlock (&self->mu_tcontrol);      
	//   std::cout<<"test3  "<<std::endl;
	listen(self->sockfd,5);
	//  std::cout<<"test4  "<<std::endl;
	self->clilen[tnum] = sizeof(self->cli_addr[tnum]);
	//	std::cout<<"witing for client to connect"<<tnum<<std::endl;
	
	
	self->newsockfd[tnum] = accept(self->sockfd,(struct sockaddr *) &self->cli_addr[tnum],&self->clilen[tnum]);
	//std::cout<<"accepting "<<tnum<<std::endl;
	if (self->newsockfd[tnum] < 0)   {
	  //  std::cout<<"Failed to accept"<<std::endl;
	}    
	else {
	  // std::cout<<"starting comunication thread client "<<tnum<<std::endl;

	  pthread_mutex_lock (&self->mu_tcontrol);
	  self->tcontrol[tnum]=1;
	  pthread_mutex_unlock (&self->mu_tcontrol);
	  
	  input *args=new input;
	  args->tnum=new int(tnum);
	  args->self=self;
	  pthread_create (&self->threads[tnum], NULL, SocketCom::ListenThread, (void*)args);
	  tnum++;
	}
	
      }
      pthread_mutex_unlock (&self->mu_tcontrol); 
    }
    pthread_mutex_unlock (&self->mu_tcontrol);
  }
  pthread_mutex_unlock (&self->mu_tcontrol);
  // std::cout<<"thread 0 exiting"<<std::endl;
  pthread_exit(NULL);
  
}

void *SocketCom::ListenThread(void* arg){
  // std::cout<<"in listen tread"<<std::endl;
  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS,NULL);
  // std::cout<<"cancel type"<<std::endl;
  input* args=(input*)arg;
  int* tnum = args->tnum;
  SocketCom* self = args->self;
  // std::cout<<"asigned stuff"<<std::endl;
  pthread_mutex_lock (&self->mu_tcontrol);
  // std::cout<<"locked control"<<std::endl;
  // std::cout<<"tum"<<tnum<<std::endl;
  // std::cout<<"*tnum"<<*tnum<<std::endl;
  while(self->tcontrol[*tnum]>0){
    //  std::cout<<"in clontrol loop"<<std::endl;
    pthread_mutex_unlock (&self->mu_tcontrol);
    
    pthread_mutex_lock (&self->mu_newbuff);
    while(!*(self->m_newbuff)){
      //  std::cout<<"in no message loop"<<std::endl;

      //bzero(self->m_buffer,256);
       memset(self->m_buffer, 0, sizeof(self->m_buffer));
      //  std::cout<<"in no message loop2"<<std::endl;
      self->n = read(self->newsockfd[*tnum],self->m_buffer,255);
      // std::cout<<"in no message loop3"<<std::endl;
      //  std::cout<<"n= "<<self->n<<std::endl;
      if (self->n < 0){  std::cout<<"Failed to read from socket"<<std::endl;
      }
      else if (self->n > 0){
	//printf("Here is the message: %s\n",self->m_buffer);
	//	std::cout<<"in no message loop4"<<std::endl;
	self->n = write(self->newsockfd[*tnum],"I got your message",18);
	//	std::cout<<"in no message loop5"<<std::endl;
	if (self->n < 0)  std::cout<<"Failed to write to socket"<<std::endl;
	//	std::cout<<"in no message loop6"<<std::endl;
	*(self->m_newbuff)=true;
      }
      else{
	pthread_mutex_lock (&self->mu_tcontrol);
	(self->tcontrol)[*tnum]=0;
	pthread_mutex_unlock (&self->mu_tcontrol);
	*(self->m_newbuff)=true;
      }
      // pthread_mutex_unlock (&self->mu_newbuff);
    }
    pthread_mutex_unlock (&self->mu_newbuff);
    
  }
  // std::cout<<"out of while"<<std::endl;
  (self->tcontrol)[*tnum]=-1;
  //std::cout<<"empty thread"<<std::endl;
  pthread_mutex_unlock (&self->mu_tcontrol);
  //std::cout<<"thread "<<*tnum<<" exiting"<<std::endl;
  close(self->newsockfd[*tnum]);
  pthread_exit(NULL);
  
}

bool SocketCom::Connect(){

  bcopy((char *)server->h_addr,(char *)&serv_addr.sin_addr.s_addr,server->h_length);
  n=connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr));
  
  if(n < 0){
  std::cout<<"Failed to connect"<<std::endl;
  Disconnect();
  return false;
  }

  return true;
  
}


bool SocketCom::Disconnect(){

 close(sockfd);

 return true;
}


bool SocketCom::Send(){

  // std::cout<<"n= "<<n<<std::endl;
   //   bzero(m_buffer,256);
   //std::cout<<"debug 1"<<std::endl;
    //fgets(m_buffer,255,stdin);
  //std::cout<<"debug 2"<<std::endl;
    n = write(sockfd,m_buffer,strlen(m_buffer));
    // std::cout<<"debug 3"<<std::endl;
    if (n < 0){
      std::cout<<"Failed writing to socket"<<std::endl;
      Disconnect();
      return false;
    }
    //std::cout<<"debug 4"<<std::endl;
    memset(m_buffer, '0', sizeof(*m_buffer));
    //memset(test,'0',sizeof(test));
    //bzero(m_buffer,256);
    //std::cout<<"debug 5"<<std::endl;
    
    //m_buffer = new char[256];
    n = read(sockfd,m_buffer,255);
    //std::cout<<"debug 6"<<std::endl;
    
    if (n < 0) {
      std::cout<<"Failed to read from socket"<<std::endl;
      Disconnect();
      return false;
    }
    //std::cout<<"debug 7"<<std::endl;
    
    //printf("%s\n",m_buffer);
    

    /*if (m_buffer != "I got your message"){
      std::cout<<"Failed to get receve confirmation"<<std::endl;
      Disconnect();
      return false;
      }*/
    //std::cout<<"debug 8"<<std::endl;
    return true;
    
}


