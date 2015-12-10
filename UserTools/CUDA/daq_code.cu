//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <sys/time.h>
#include "helper_cuda.h"
#include "library_daq.h"


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <thrust/extrema.h>

/////////////////////////////
// define global variables //
/////////////////////////////
/// parameters
double distance_between_vertices; // linear distance between test vertices
double wall_like_distance; // distance from wall (in units of distance_between_vertices) to define wall-like events
unsigned int time_step_size; // time binning for the trigger
__constant__ unsigned int constant_time_step_size; 
unsigned int water_like_threshold_number_of_pmts; // number of pmts above which a trigger is possible for water-like events
unsigned int wall_like_threshold_number_of_pmts; // number of pmts above which a trigger is possible for wall-like events
double coalesce_time; // time such that if two triggers are closer than this they are coalesced into a single trigger
double trigger_gate_up; // duration to be saved after the trigger time
double trigger_gate_down; // duration to be saved before the trigger time
/// detector
double detector_height; // detector height
double detector_radius; // detector radius
/// pmts
unsigned int n_PMTs; // number of pmts in the detector
__constant__ unsigned int constant_n_PMTs;
double * PMT_x, *PMT_y, *PMT_z; // coordinates of the pmts in the detector
/// vertices
unsigned int n_test_vertices; // number of test vertices
unsigned int n_water_like_test_vertices; // number of test vertices
__constant__ unsigned int constant_n_test_vertices;
__constant__ unsigned int constant_n_water_like_test_vertices;
double * vertex_x, * vertex_y, * vertex_z; // coordinates of test vertices
/// threads
unsigned int number_of_kernel_blocks;  // number of cores to be used
dim3 number_of_kernel_blocks_3d;
unsigned int number_of_threads_per_block; // number of threads per core to be used
dim3 number_of_threads_per_block_3d;
unsigned int grid_size;  // grid = (n cores) X (n threads / core)
/// hits
double time_offset;  // ns, offset to make times positive
__constant__ double constant_time_offset;
unsigned int n_time_bins; // number of time bins 
__constant__ unsigned int constant_n_time_bins;
unsigned int n_hits; // number of input hits from the detector
__constant__ unsigned int constant_n_hits;
unsigned int * host_ids; // pmt id of a hit
unsigned int *device_ids;
texture<unsigned int, 1, cudaReadModeElementType> tex_ids;
unsigned int * host_times;  // time of a hit
unsigned int *device_times;
texture<unsigned int, 1, cudaReadModeElementType> tex_times;
// npmts per time bin
unsigned int * device_n_pmts_per_time_bin; // number of active pmt in a time bin
// tof
double speed_light_water;
float *device_times_of_flight; // time of flight between a vertex and a pmt
float *host_times_of_flight;
texture<float, 1, cudaReadModeElementType> tex_times_of_flight;
// triggers
std::vector<std::pair<unsigned int,unsigned int> > candidate_trigger_pair_vertex_time;  // pair = (v, t) = (a vertex, a time at the end of the 2nd of two coalesced bins)
std::vector<unsigned int> candidate_trigger_npmts_in_time_bin; // npmts in time bin
std::vector<std::pair<unsigned int,unsigned int> > trigger_pair_vertex_time;
std::vector<unsigned int> trigger_npmts_in_time_bin;
std::vector<std::pair<unsigned int,unsigned int> > final_trigger_pair_vertex_time;
std::vector<double> output_trigger_information;
// C timing
struct timeval t0;
struct timeval t1;
// CUDA timing
cudaEvent_t start, stop, total_start, total_stop;
// find candidates
unsigned int * host_max_number_of_pmts_in_time_bin;
unsigned int * device_max_number_of_pmts_in_time_bin;
unsigned int *  host_vertex_with_max_n_pmts;
unsigned int *  device_vertex_with_max_n_pmts;
// gpu properties
int max_n_threads_per_block;
int max_n_blocks;
// verbosity level
bool use_verbose;
// files
std::string detector_file;
std::string pmts_file;

float elapsed_parameters, elapsed_pmts, elapsed_detector, elapsed_vertices,
  elapsed_threads, elapsed_tof, elapsed_memory_tofs_dev, elapsed_memory_candidates_host, elapsed_tofs_copy_dev,
  elapsed_input, elapsed_memory_dev, elapsed_copy_dev, elapsed_kernel, 
  elapsed_threads_candidates, elapsed_candidates_memory_dev, elapsed_candidates_kernel,
  elapsed_candidates_copy_host, choose_candidates, elapsed_coalesce, elapsed_gates, elapsed_free, elapsed_total,
  elapsed_tofs_free, elapsed_reset;
bool use_timing;

__global__ void kernel_correct_times(unsigned int *ct);
__global__ void kernel_find_vertex_with_max_npmts_in_timebin(unsigned int * np, unsigned int * mnp, unsigned int * vmnp);
__device__ unsigned int device_get_distance_index(unsigned int pmt_id, unsigned int vertex_block);
__device__ unsigned int device_get_time_index(unsigned int hit_index, unsigned int vertex_block);




//
// main code
//

int CUDAFunction(std::vector<int> PMTids, std::vector<int> times)
{


  ////////////////
  // read input //
  ////////////////
  // set: n_hits, host_ids, host_times, time_offset, n_time_bins
  // use: time_offset, n_test_vertices
  // memcpy: constant_n_time_bins, constant_n_hits
  if( use_timing )
    start_c_clock();
  if( !read_the_input(PMTids, times) ) return 0;
  if( use_timing )
    elapsed_input += stop_c_clock();
  
  
  
  ////////////////////////////////////////
  // allocate candidates memory on host //
  ////////////////////////////////////////
  // use: n_time_bins, n_hits
  // malloc: host_max_number_of_pmts_in_time_bin, host_vertex_with_max_n_pmts
  if( use_timing )
    start_cuda_clock();
  allocate_candidates_memory_on_host();
  if( use_timing )
    elapsed_memory_candidates_host += stop_cuda_clock();
  
  
  ////////////////////////////////////////////////
  // set number of blocks and threads per block //
  ////////////////////////////////////////////////
  // set: number_of_kernel_blocks, number_of_threads_per_block
  // use: n_test_vertices, n_hits
  if( use_timing )
    start_c_clock();
  if( !setup_threads_for_tof_2d() ) return 0;
  if( use_timing )
    elapsed_threads += stop_c_clock();
  
  
  
  ///////////////////////////////////////
  // allocate correct memory on device //
  ///////////////////////////////////////
  // use: n_test_vertices, n_hits, n_time_bins
  // cudamalloc: device_ids, device_times, device_n_pmts_per_time_bin
  if( use_timing )
    start_cuda_clock();
  allocate_correct_memory_on_device();
  if( use_timing )
    elapsed_memory_dev += stop_cuda_clock();
  
  
  //////////////////////////////////////
  // copy input into device variables //
  //////////////////////////////////////
  // use: n_hits
  // memcpy: device_ids, device_times, constant_time_offset
  // texture: tex_ids, tex_times
  if( use_timing )
    start_cuda_clock();
  fill_correct_memory_on_device();
  if( use_timing )
    elapsed_copy_dev += stop_cuda_clock();
  
  
  
  ////////////////////
  // execute kernel //
  ////////////////////
  if( use_timing )
    start_cuda_clock();
  printf(" --- execute kernel \n");
  kernel_correct_times<<<number_of_kernel_blocks_3d,number_of_threads_per_block_3d>>>(device_n_pmts_per_time_bin);
  getLastCudaError("correct_kernel execution failed\n");
  if( use_timing )
    elapsed_kernel += stop_cuda_clock();
  
  
  
  //////////////////////////////////
  // setup threads for candidates //
  //////////////////////////////////
  // set: number_of_kernel_blocks, number_of_threads_per_block
  // use: n_time_bins
  if( use_timing )
    start_c_clock();
  if( !setup_threads_to_find_candidates() ) return 0;
  if( use_timing )
    elapsed_threads_candidates += stop_c_clock();
  
  
  
  //////////////////////////////////////////
  // allocate candidates memory on device //
  //////////////////////////////////////////
  // use: n_time_bins
  // cudamalloc: device_max_number_of_pmts_in_time_bin, device_vertex_with_max_n_pmts
  if( use_timing )
    start_cuda_clock();
  allocate_candidates_memory_on_device();
  if( use_timing )
    elapsed_candidates_memory_dev += stop_cuda_clock();
  
  
  
  /////////////////////////////////////
  // find candidates above threshold //
  /////////////////////////////////////
  if( use_timing )
    start_cuda_clock();
  if( use_verbose )
    printf(" --- execute candidates kernel \n");
  kernel_find_vertex_with_max_npmts_in_timebin<<<number_of_kernel_blocks,number_of_threads_per_block>>>(device_n_pmts_per_time_bin, device_max_number_of_pmts_in_time_bin, device_vertex_with_max_n_pmts);
  getLastCudaError("candidates_kernel execution failed\n");
  if( use_timing )
    elapsed_candidates_kernel += stop_cuda_clock();
  
  
  
  
  /////////////////////////////////////////
  // copy candidates from device to host //
  /////////////////////////////////////////
  // use: n_time_bins
  // memcpy: host_max_number_of_pmts_in_time_bin, host_vertex_with_max_n_pmts
  if( use_timing )
    start_cuda_clock();
  if( use_verbose )
    printf(" --- copy candidates from device to host \n");
  copy_candidates_from_device_to_host();
  if( use_timing )
    elapsed_candidates_copy_host += stop_cuda_clock();
  

  
  ///////////////////////////////////////
  // choose candidates above threshold //
  ///////////////////////////////////////
  if( use_timing )
    start_cuda_clock();
  if( use_verbose )
    printf(" --- choose candidates above threshold \n");
  choose_candidates_above_threshold();
  if( use_timing )
    choose_candidates = stop_cuda_clock();
  
  
  
  ///////////////////////
  // coalesce triggers //
  ///////////////////////
  if( use_timing )
    start_cuda_clock();
  coalesce_triggers();
  if( use_timing )
    elapsed_coalesce += stop_cuda_clock();
  
  
  
  
  //////////////////////////////////
  // separate triggers into gates //
  //////////////////////////////////
  if( use_timing )
    start_cuda_clock();
  separate_triggers_into_gates();
  if( use_timing )
    elapsed_gates += stop_cuda_clock();
  

  int the_output = trigger_pair_vertex_time.size(); 
  
  
  /////////////////////////////
  // deallocate event memory //
  /////////////////////////////
  if( use_timing )
    start_cuda_clock();
  if( use_verbose )
    printf(" --- deallocate memory \n");
  free_event_memories();
  if( use_timing )
    elapsed_free += stop_cuda_clock();


  printf(" ------ analyzed event \n");

  return the_output;
}



//
// kernel routine
// 

// __global__ identifier says it's a kernel function
__global__ void kernel_correct_times(unsigned int *ct){

  // get unique id for each thread in each block
  unsigned int tid_x = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int tid_y = threadIdx.y + blockDim.y*blockIdx.y;

  // map the two 2D indices to a single linear, 1D index
  int tid = tid_y * gridDim.x * blockDim.x + tid_x;

  // tid runs from 0 to n_test_vertices * n_hits:
  //      vertex 0           vertex 1       ...     vertex m
  // (hit 0, ..., hit n; hit 0, ..., hit n; ...; hit 0, ..., hit n);

  unsigned int vertex_index = (int)(tid/constant_n_hits);
  unsigned int hit_index = tid % constant_n_hits;

  //  printf(" tid %d tidx %d tidy %d v %d h %d \n", tid, tid_x, tid_y, vertex_index, hit_index);

  //    printf( " threadi %d blockdim %d blockid %d, tid %d, vertex_index %d, hit %d \n",
  //  	  threadIdx.x, blockDim.x, blockIdx.x, tid,
  //	  vertex_index, hit_index);

  // skip if thread is assigned to nonexistent vertex
  if( vertex_index >= constant_n_test_vertices ) return;

  // skip if thread is assigned to nonexistent hit
  if( hit_index >= constant_n_hits ) return;

  unsigned int vertex_block = constant_n_time_bins*vertex_index;

  unsigned int vertex_block2 = constant_n_PMTs*vertex_index;

  atomicAdd(&
	    ct[
	       device_get_time_index(
				     int(floor(
					       (tex1Dfetch(tex_times,hit_index)
						- tex1Dfetch(tex_times_of_flight,
							     device_get_distance_index(
										       tex1Dfetch(tex_ids,hit_index),
										       vertex_block2
										       )
							     )
						+ constant_time_offset)/constant_time_step_size
					       )
					 ),
				     vertex_block
				     )
	       ]
	    ,1);

  //  printf( " hit %d (nh %d) id %d t %d; vertex %d (nv %d) tof %f  %d \n", hit_index, constant_n_hits, ids[hit_index], t[hit_index], vertex_index, constant_n_test_vertices, tof, ct[time_index]);

  return;

}




bool read_input(std::vector<int> PMTids, std::vector<int> times, int * max_time){

  int time;
  int min = INT_MAX;
  int max = INT_MIN;
  for( unsigned int i=0; i<n_hits; i++){
    time = times[i];
    host_times[i] = times[i];
    host_ids[i] = PMTids[i];
    if( time > max ) max = time;
    if( time < min ) min = time;
  }

  if( min < 0 ){
    time_offset -= min;
  }


  *max_time = max;

  return true;

}


bool read_detector(){

  FILE *f=fopen(detector_file.c_str(), "r");
  double pmt_radius;
  if( fscanf(f, "%lf %lf %lf", &detector_height, &detector_radius, &pmt_radius) != 3 ){
    printf(" problem scanning detector \n");
    fclose(f);
    return false;
  }

  fclose(f);
  return true;

}



void print_parameters(){

  printf(" n_test_vertices = %d \n", n_test_vertices);
  printf(" n_water_like_test_vertices = %d \n", n_water_like_test_vertices);
  printf(" n_PMTs = %d \n", n_PMTs);
  printf(" number_of_kernel_blocks = %d \n", number_of_kernel_blocks);
  printf(" number_of_threads_per_block = %d \n", number_of_threads_per_block);
  printf(" grid size = %d -> %d \n", number_of_kernel_blocks*number_of_threads_per_block, grid_size);

}

void print_parameters_2d(){

  printf(" n_test_vertices = %d \n", n_test_vertices);
  printf(" n_water_like_test_vertices = %d \n", n_water_like_test_vertices);
  printf(" n_PMTs = %d \n", n_PMTs);
  printf(" number_of_kernel_blocks = (%d, %d) = %d \n", number_of_kernel_blocks_3d.x, number_of_kernel_blocks_3d.y, number_of_kernel_blocks_3d.x * number_of_kernel_blocks_3d.y);
  printf(" number_of_threads_per_block = (%d, %d) = %d \n", number_of_threads_per_block_3d.x, number_of_threads_per_block_3d.y, number_of_threads_per_block_3d.x * number_of_threads_per_block_3d.y);
  printf(" grid size = %d -> %d \n", number_of_kernel_blocks_3d.x*number_of_kernel_blocks_3d.y*number_of_threads_per_block_3d.x*number_of_threads_per_block_3d.y, grid_size);

}

void print_input(){

  for(unsigned int i=0; i<n_hits; i++)
    printf(" hit %d time %d id %d \n", i, host_times[i], host_ids[i]);

}

void print_pmts(){

  for(unsigned int i=0; i<n_PMTs; i++)
    printf(" pmt %d x %f y %f z %f  \n", i, PMT_x[i], PMT_y[i], PMT_z[i]);

}

void print_times_of_flight(){

  printf(" times_of_flight: (vertex, PMT) \n");
  unsigned int distance_index;
  for(unsigned int iv=0; iv<n_test_vertices; iv++){
    printf(" ( ");
    for(unsigned int ip=0; ip<n_PMTs; ip++){
      distance_index = get_distance_index(host_ids[ip], n_PMTs*iv);
      printf(" %f ", host_times_of_flight[distance_index]);
    }
    printf(" ) \n");
  }
}


bool read_the_pmts(){

  printf(" --- read pmts \n");
  n_PMTs = read_number_of_pmts();
  if( !n_PMTs ) return false;
  printf(" detector contains %d PMTs \n", n_PMTs);
  PMT_x = (double *)malloc(n_PMTs*sizeof(double));
  PMT_y = (double *)malloc(n_PMTs*sizeof(double));
  PMT_z = (double *)malloc(n_PMTs*sizeof(double));
  if( !read_pmts() ) return false;
  //print_pmts();
  return true;

}

bool read_the_detector(){

  printf(" --- read detector \n");
  if( !read_detector() ) return false;
  printf(" detector height %f cm, radius %f cm \n", detector_height, detector_radius);
  return true;

}

void make_test_vertices(){

  printf(" --- make test vertices \n");
  float semiheight = detector_height/2.;
  n_test_vertices = 0;
  // 1: count number of test vertices
  for(int i=-1*semiheight; i <= semiheight; i+=distance_between_vertices) {
    for(int j=-1*detector_radius; j<=detector_radius; j+=distance_between_vertices) {
      for(int k=-1*detector_radius; k<=detector_radius; k+=distance_between_vertices) {
	if(pow(j,2)+pow(k,2) > pow(detector_radius,2))
	  continue;
	n_test_vertices++;
      }
    }
  }
  vertex_x = (double *)malloc(n_test_vertices*sizeof(double));
  vertex_y = (double *)malloc(n_test_vertices*sizeof(double));
  vertex_z = (double *)malloc(n_test_vertices*sizeof(double));

  // 2: assign coordinates to test vertices
  // water-like events
  n_test_vertices = 0;
  for(int i=-1*semiheight; i <= semiheight; i+=distance_between_vertices) {
    for(int j=-1*detector_radius; j<=detector_radius; j+=distance_between_vertices) {
      for(int k=-1*detector_radius; k<=detector_radius; k+=distance_between_vertices) {

	
	if( 
	   // skip endcap region
	   abs(i) > semiheight - wall_like_distance*distance_between_vertices ||
	   // skip sidewall region
	   pow(j,2)+pow(k,2) > pow(detector_radius - wall_like_distance*distance_between_vertices,2)
	    ) continue;
	
	vertex_x[n_test_vertices] = j*1.;
	vertex_y[n_test_vertices] = k*1.;
	vertex_z[n_test_vertices] = i*1.;
	n_test_vertices++;
      }
    }
  }
  n_water_like_test_vertices = n_test_vertices;

  // wall-like events
  for(int i=-1*semiheight; i <= semiheight; i+=distance_between_vertices) {
    for(int j=-1*detector_radius; j<=detector_radius; j+=distance_between_vertices) {
      for(int k=-1*detector_radius; k<=detector_radius; k+=distance_between_vertices) {

	if( 
	   abs(i) > semiheight - wall_like_distance*distance_between_vertices ||
	   pow(j,2)+pow(k,2) > pow(detector_radius - wall_like_distance*distance_between_vertices,2)
	    ){

	  if(pow(j,2)+pow(k,2) > pow(detector_radius,2)) continue;
	  
	  vertex_x[n_test_vertices] = j*1.;
	  vertex_y[n_test_vertices] = k*1.;
	  vertex_z[n_test_vertices] = i*1.;
	  n_test_vertices++;
	}
      }
    }
  }

  return;

}

bool setup_threads_for_tof(){

  grid_size = n_test_vertices;

  number_of_kernel_blocks = grid_size / max_n_threads_per_block + 1;
  number_of_threads_per_block = ( number_of_kernel_blocks > 1 ? max_n_threads_per_block : grid_size);

  print_parameters();

  if( number_of_threads_per_block > max_n_threads_per_block ){
    printf(" warning: number_of_threads_per_block = %d cannot exceed max value %d \n", number_of_threads_per_block, max_n_threads_per_block );
    return false;
  }

  if( number_of_kernel_blocks > max_n_blocks ){
    printf(" warning: number_of_kernel_blocks = %d cannot exceed max value %d \n", number_of_kernel_blocks, max_n_blocks );
    return false;
  }

  return true;
}


bool setup_threads_for_tof_biparallel(){

  grid_size = n_test_vertices * n_hits;

  number_of_kernel_blocks = grid_size / max_n_threads_per_block + 1;
  number_of_threads_per_block = ( number_of_kernel_blocks > 1 ? max_n_threads_per_block : grid_size);

  print_parameters();

  if( number_of_threads_per_block > max_n_threads_per_block ){
    printf(" --------------------- warning: number_of_threads_per_block = %d cannot exceed max value %d \n", number_of_threads_per_block, max_n_threads_per_block );
    return false;
  }

  if( number_of_kernel_blocks > max_n_blocks ){
    printf(" warning: number_of_kernel_blocks = %d cannot exceed max value %d \n", number_of_kernel_blocks, max_n_blocks );
    return false;
  }

  return true;

}

bool setup_threads_for_tof_2d(){

  grid_size = n_test_vertices * n_hits;
  unsigned int max_n_threads_per_block_2d = sqrt(max_n_threads_per_block);

  number_of_kernel_blocks_3d.x = n_test_vertices / max_n_threads_per_block_2d + 1;
  number_of_kernel_blocks_3d.y = n_hits / max_n_threads_per_block_2d + 1;

  number_of_threads_per_block_3d.x = ( number_of_kernel_blocks_3d.x > 1 ? max_n_threads_per_block_2d : n_test_vertices);
  number_of_threads_per_block_3d.y = ( number_of_kernel_blocks_3d.y > 1 ? max_n_threads_per_block_2d : n_hits);

  print_parameters_2d();

  if( number_of_threads_per_block_3d.x > max_n_threads_per_block_2d ){
    printf(" --------------------- warning: number_of_threads_per_block x = %d cannot exceed max value %d \n", number_of_threads_per_block_3d.x, max_n_threads_per_block_2d );
    return false;
  }

  if( number_of_threads_per_block_3d.y > max_n_threads_per_block_2d ){
    printf(" --------------------- warning: number_of_threads_per_block y = %d cannot exceed max value %d \n", number_of_threads_per_block_3d.y, max_n_threads_per_block_2d );
    return false;
  }

  if( number_of_kernel_blocks_3d.x > max_n_blocks ){
    printf(" warning: number_of_kernel_blocks x = %d cannot exceed max value %d \n", number_of_kernel_blocks_3d.x, max_n_blocks );
    return false;
  }

  if( number_of_kernel_blocks_3d.y > max_n_blocks ){
    printf(" warning: number_of_kernel_blocks y = %d cannot exceed max value %d \n", number_of_kernel_blocks_3d.y, max_n_blocks );
    return false;
  }

  return true;

}

bool setup_threads_to_find_candidates(){

  number_of_kernel_blocks = n_time_bins / max_n_threads_per_block + 1;
  number_of_threads_per_block = ( number_of_kernel_blocks > 1 ? max_n_threads_per_block : n_time_bins);

  if( number_of_threads_per_block > max_n_threads_per_block ){
    printf(" warning: number_of_threads_per_block = %d cannot exceed max value %d \n", number_of_threads_per_block, max_n_threads_per_block );
    return false;
  }

  return true;
}


bool read_the_input(std::vector<int> PMTids, std::vector<int> times){

  printf(" --- read input \n");
  n_hits = PMTids.size();
  if( !n_hits ) return false;
  printf(" input contains %d hits \n", n_hits);
  host_ids = (unsigned int *)malloc(n_hits*sizeof(unsigned int));
  host_times = (unsigned int *)malloc(n_hits*sizeof(unsigned int));
  int max_time;
  if( !read_input(PMTids, times, &max_time) ) return false;
  //time_offset = 600.; // set to constant to match trevor running
  n_time_bins = int(floor((max_time + time_offset)/time_step_size))+1; // floor returns the integer below
  printf(" input max_time %d, n_time_bins %d \n", max_time, n_time_bins);
  printf(" time_offset = %f ns \n", time_offset);
  //print_input();

  checkCudaErrors( cudaMemcpyToSymbol(constant_n_time_bins, &n_time_bins, sizeof(n_time_bins)) );
  checkCudaErrors( cudaMemcpyToSymbol(constant_n_hits, &n_hits, sizeof(n_hits)) );

  return true;
}

void allocate_tofs_memory_on_device(){

  printf(" --- allocate memory tofs \n");
  checkCudaErrors(cudaMalloc((void **)&device_times_of_flight, n_test_vertices*n_PMTs*sizeof(float)));
  /*
    if( n_hits*n_test_vertices > available_memory ){
    printf(" cannot allocate vector of %d, available_memory %d \n", n_hits*n_test_vertices, available_memory);
    return 0;
    }
  */


  return;

}

void allocate_correct_memory_on_device(){

  printf(" --- allocate memory \n");
  /*
    if( n_hits > available_memory ){
    printf(" cannot allocate vector of %d, available_memory %d \n", n_hits, available_memory);
    return 0;
    }
  */
  checkCudaErrors(cudaMalloc((void **)&device_ids, n_hits*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **)&device_times, n_hits*sizeof(unsigned int)));
  /*
    if( n_test_vertices*n_PMTs > available_memory ){
    printf(" cannot allocate vector of %d, available_memory %d \n", n_test_vertices*n_PMTs, available_memory);
    return 0;
    }
  */
  checkCudaErrors(cudaMalloc((void **)&device_n_pmts_per_time_bin, n_time_bins*n_test_vertices*sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(device_n_pmts_per_time_bin, 0, n_time_bins*n_test_vertices*sizeof(unsigned int)));

  return;

}

void allocate_candidates_memory_on_host(){

  printf(" --- allocate candidates memory on host \n");

  host_max_number_of_pmts_in_time_bin = (unsigned int *)malloc(n_time_bins*sizeof(unsigned int));
  host_vertex_with_max_n_pmts = (unsigned int *)malloc(n_time_bins*sizeof(unsigned int));

  return;

}

void allocate_candidates_memory_on_device(){

  printf(" --- allocate candidates memory on device \n");

  checkCudaErrors(cudaMalloc((void **)&device_max_number_of_pmts_in_time_bin, n_time_bins*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void **)&device_vertex_with_max_n_pmts, n_time_bins*sizeof(unsigned int)));

  return;

}

void make_table_of_tofs(){

  printf(" --- fill times_of_flight \n");
  host_times_of_flight = (float*)malloc(n_test_vertices*n_PMTs * sizeof(double));
  printf(" speed_light_water %f \n", speed_light_water);
  unsigned int distance_index;
  time_offset = 0.;
  for(unsigned int ip=0; ip<n_PMTs; ip++){
    for(unsigned int iv=0; iv<n_test_vertices; iv++){
      distance_index = get_distance_index(ip + 1, n_PMTs*iv);
      host_times_of_flight[distance_index] = sqrt(pow(vertex_x[iv] - PMT_x[ip],2) + pow(vertex_y[iv] - PMT_y[ip],2) + pow(vertex_z[iv] - PMT_z[ip],2))/speed_light_water;
      if( host_times_of_flight[distance_index] > time_offset )
	time_offset = host_times_of_flight[distance_index];

    }
  }
  //print_times_of_flight();

  return;
}


void fill_tofs_memory_on_device(){

  printf(" --- copy tofs from host to device \n");
  checkCudaErrors(cudaMemcpy(device_times_of_flight,
			     host_times_of_flight,
			     n_test_vertices*n_PMTs*sizeof(float),
			     cudaMemcpyHostToDevice));
  checkCudaErrors( cudaMemcpyToSymbol(constant_time_step_size, &time_step_size, sizeof(time_step_size)) );
  checkCudaErrors( cudaMemcpyToSymbol(constant_n_test_vertices, &n_test_vertices, sizeof(n_test_vertices)) );
  checkCudaErrors( cudaMemcpyToSymbol(constant_n_water_like_test_vertices, &n_water_like_test_vertices, sizeof(n_water_like_test_vertices)) );
  checkCudaErrors( cudaMemcpyToSymbol(constant_n_PMTs, &n_PMTs, sizeof(n_PMTs)) );

  // Bind the array to the texture
  checkCudaErrors(cudaBindTexture(0,tex_times_of_flight, device_times_of_flight, n_test_vertices*n_PMTs*sizeof(float)));
  


  return;
}


void fill_correct_memory_on_device(){

  printf(" --- copy from host to device \n");
  checkCudaErrors(cudaMemcpy(device_ids,
			     host_ids,
			     n_hits*sizeof(unsigned int),
			     cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_times,
			     host_times,
			     n_hits*sizeof(unsigned int),
			     cudaMemcpyHostToDevice));
  checkCudaErrors( cudaMemcpyToSymbol(constant_time_offset, &time_offset, sizeof(time_offset)) );

  checkCudaErrors(cudaBindTexture(0,tex_ids, device_ids, n_hits*sizeof(unsigned int)));
  checkCudaErrors(cudaBindTexture(0,tex_times, device_times, n_hits*sizeof(unsigned int)));


  return;
}





unsigned int read_number_of_pmts(){

  FILE *f=fopen(pmts_file.c_str(), "r");
  if (f == NULL){
    printf(" cannot read pmts file \n");
    fclose(f);
    return 0;
  }

  unsigned int n_pmts = 0;

  for (char c = getc(f); c != EOF; c = getc(f))
    if (c == '\n')
      n_pmts ++;

  fclose(f);
  return n_pmts;

}

bool read_pmts(){

  FILE *f=fopen(pmts_file.c_str(), "r");

  double x, y, z;
  unsigned int id;
  for( unsigned int i=0; i<n_PMTs; i++){
    if( fscanf(f, "%d %lf %lf %lf", &id, &x, &y, &z) != 4 ){
      printf(" problem scanning pmt %d \n", i);
      fclose(f);
      return false;
    }
    PMT_x[id-1] = x;
    PMT_y[id-1] = y;
    PMT_z[id-1] = z;
  }

  fclose(f);
  return true;

}


void coalesce_triggers(){

  trigger_pair_vertex_time.clear();
  trigger_npmts_in_time_bin.clear();

  unsigned int vertex_index, time_lower, time_upper, number_of_pmts_in_time_bin;
  unsigned int vertex_index_prev=0, time_lower_prev=0, time_upper_prev=0, number_of_pmts_in_time_bin_prev=0;
  unsigned int max_pmt=0,max_vertex_index=0,max_time=0;
  bool first_trigger, last_trigger, coalesce_triggers;
  unsigned int trigger_index;
  for(std::vector<std::pair<unsigned int,unsigned int> >::const_iterator itrigger=candidate_trigger_pair_vertex_time.begin(); itrigger != candidate_trigger_pair_vertex_time.end(); ++itrigger){

    vertex_index =      itrigger->first;
    time_upper = itrigger->second;
    time_lower = time_upper-1;
    trigger_index = itrigger - candidate_trigger_pair_vertex_time.begin();
    number_of_pmts_in_time_bin = candidate_trigger_npmts_in_time_bin.at(trigger_index);

    first_trigger = (trigger_index == 0);
    last_trigger = (trigger_index == candidate_trigger_pair_vertex_time.size()-1);
       
    if( first_trigger ){
      if( number_of_pmts_in_time_bin > 0){
	max_pmt = number_of_pmts_in_time_bin;
	max_vertex_index = vertex_index;
	max_time = time_upper;
      }
    }
    else{
      coalesce_triggers = (std::abs((int)(max_time - time_upper)) < coalesce_time/time_step_size);

      if( coalesce_triggers ){
	if( number_of_pmts_in_time_bin > max_pmt) {
	  max_pmt = number_of_pmts_in_time_bin;
	  max_vertex_index = vertex_index;
	  max_time = time_upper;
	}
      }else{
	trigger_pair_vertex_time.push_back(std::make_pair(max_vertex_index,max_time));
	trigger_npmts_in_time_bin.push_back(max_pmt);
	max_pmt = number_of_pmts_in_time_bin;
	max_vertex_index = vertex_index;
	max_time = time_upper;     
      }
    }

    if(last_trigger){
      trigger_pair_vertex_time.push_back(std::make_pair(max_vertex_index,max_time));
      trigger_npmts_in_time_bin.push_back(max_pmt);
    }
     
    time_upper_prev = time_upper;
    time_lower_prev = time_lower;
    vertex_index_prev = vertex_index; 
    number_of_pmts_in_time_bin_prev = number_of_pmts_in_time_bin;
  }

  for(std::vector<std::pair<unsigned int,unsigned int> >::const_iterator itrigger=trigger_pair_vertex_time.begin(); itrigger != trigger_pair_vertex_time.end(); ++itrigger)
    printf(" coalesced trigger timebin %d vertex index %d \n", itrigger->first, itrigger->second);

  return;

}


void separate_triggers_into_gates(){

  final_trigger_pair_vertex_time.clear();
  unsigned int trigger_index;

  unsigned int time_start=0;
  for(std::vector<std::pair<unsigned int,unsigned int> >::const_iterator itrigger=trigger_pair_vertex_time.begin(); itrigger != trigger_pair_vertex_time.end(); ++itrigger){
    //once a trigger is found, we must jump in the future before searching for the next
    if(itrigger->second > time_start) {
      unsigned int triggertime = itrigger->second*time_step_size - time_offset;
      final_trigger_pair_vertex_time.push_back(std::make_pair(itrigger->first,triggertime));
      time_start = triggertime + trigger_gate_up;
      trigger_index = itrigger - trigger_pair_vertex_time.begin();
      output_trigger_information.clear();
      output_trigger_information.push_back(vertex_x[itrigger->first]);
      output_trigger_information.push_back(vertex_y[itrigger->first]);
      output_trigger_information.push_back(vertex_z[itrigger->first]);
      output_trigger_information.push_back(trigger_npmts_in_time_bin.at(trigger_index));
      output_trigger_information.push_back(triggertime);

      printf(" triggertime: %d, npmts: %d, x: %f, y: %f, z: %f \n", triggertime, trigger_npmts_in_time_bin.at(trigger_index), vertex_x[itrigger->first], vertex_y[itrigger->first], vertex_z[itrigger->first]);


    }
  }


  return;
}


float timedifference_msec(struct timeval t0, struct timeval t1){
  return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}



void start_c_clock(){
  gettimeofday(&t0,0);

}
double stop_c_clock(){
  gettimeofday(&t1,0);
  return timedifference_msec(t0, t1);
}
void start_cuda_clock(){
  cudaEventRecord(start);

}
double stop_cuda_clock(){
  float milli;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);
  return milli;
}
void start_total_cuda_clock(){
  cudaEventRecord(total_start);

}
double stop_total_cuda_clock(){
  float milli;
  cudaEventRecord(total_stop);
  cudaEventSynchronize(total_stop);
  cudaEventElapsedTime(&milli, total_start, total_stop);
  return milli;
}

unsigned int get_distance_index(unsigned int pmt_id, unsigned int vertex_block){
  // block = (npmts) * (vertex index)

  return pmt_id - 1 + vertex_block;

}

unsigned int get_time_index(unsigned int hit_index, unsigned int vertex_block){
  // block = (n time bins) * (vertex index)

  return hit_index + vertex_block;

}

__device__ unsigned int device_get_distance_index(unsigned int pmt_id, unsigned int vertex_block){
  // block = (npmts) * (vertex index)

  return pmt_id - 1 + vertex_block;

}

__device__ unsigned int device_get_time_index(unsigned int hit_index, unsigned int vertex_block){
  // block = (n time bins) * (vertex index)

  return hit_index + vertex_block;

}

// Print device properties
void print_gpu_properties(){

  int devCount;
  cudaGetDeviceCount(&devCount);
  printf(" CUDA Device Query...\n");
  printf(" There are %d CUDA devices.\n", devCount);
  cudaDeviceProp devProp;
  for (int i = 0; i < devCount; ++i){
    // Get device properties
    printf(" CUDA Device #%d\n", i);
    cudaGetDeviceProperties(&devProp, i);
    printf("Major revision number:          %d\n",  devProp.major);
    printf("Minor revision number:          %d\n",  devProp.minor);
    printf("Name:                           %s\n",  devProp.name);
    printf("Total global memory:            %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block:  %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:      %d\n",  devProp.regsPerBlock);
    printf("Warp size:                      %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:           %lu\n",  devProp.memPitch);
    max_n_threads_per_block = devProp.maxThreadsPerBlock;
    printf("Maximum threads per block:      %d\n",  max_n_threads_per_block);
    for (int i = 0; i < 3; ++i)
      printf("Maximum dimension %d of block:   %d\n", i, devProp.maxThreadsDim[i]);
    max_n_blocks = devProp.maxGridSize[0];
    for (int i = 0; i < 3; ++i)
      printf("Maximum dimension %d of grid:    %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                     %d\n",  devProp.clockRate);
    printf("Total constant memory:          %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:              %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution:  %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:      %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:       %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
    printf("Memory Clock Rate (KHz):        %d\n", devProp.memoryClockRate);
    printf("Memory Bus Width (bits):        %d\n", devProp.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s):   %f\n", 2.0*devProp.memoryClockRate*(devProp.memoryBusWidth/8)/1.0e6);
    printf("Concurrent kernels:             %d\n",  devProp.concurrentKernels);
  }
  size_t available_memory, total_memory;
  cudaMemGetInfo(&available_memory, &total_memory);
  size_t stack_memory;
  cudaDeviceGetLimit(&stack_memory, cudaLimitStackSize);
  size_t fifo_memory;
  cudaDeviceGetLimit(&fifo_memory, cudaLimitPrintfFifoSize);
  size_t heap_memory;
  cudaDeviceGetLimit(&heap_memory, cudaLimitMallocHeapSize);
  printf(" memgetinfo: available_memory %f MB, total_memory %f MB, stack_memory %f MB, fifo_memory %f MB, heap_memory %f MB \n", (double)available_memory/1.e6, (double)total_memory/1.e6, (double)stack_memory/1.e6, (double)fifo_memory/1.e6, (double)heap_memory/1.e6);


  return;
}


__global__ void kernel_find_vertex_with_max_npmts_in_timebin(unsigned int * np, unsigned int * mnp, unsigned int * vmnp){


  // get unique id for each thread in each block == time bin
  unsigned int time_bin_index = threadIdx.x + blockDim.x*blockIdx.x;

  // skip if thread is assigned to nonexistent time bin
  if( time_bin_index >= constant_n_time_bins ) return;


  unsigned int number_of_pmts_in_time_bin = 0;
  unsigned int time_index;
  unsigned int max_number_of_pmts_in_time_bin=0;
  unsigned int vertex_with_max_n_pmts = 0;

  for(unsigned int iv=0;iv<constant_n_test_vertices;iv++) { // loop over test vertices
    // sum the number of hit PMTs in this time window
    
    time_index = time_bin_index + constant_n_time_bins*iv;
    if( time_index >= constant_n_time_bins*constant_n_test_vertices - 1 ) continue;
    number_of_pmts_in_time_bin = np[time_index] + np[time_index+1];
    if( number_of_pmts_in_time_bin > max_number_of_pmts_in_time_bin ){
      max_number_of_pmts_in_time_bin = number_of_pmts_in_time_bin;
      vertex_with_max_n_pmts = iv;
    }
  }

  mnp[time_bin_index] = max_number_of_pmts_in_time_bin;
  vmnp[time_bin_index] = vertex_with_max_n_pmts;

  return;

}

void free_event_memories(){

  checkCudaErrors(cudaUnbindTexture(tex_ids));
  checkCudaErrors(cudaUnbindTexture(tex_times));
  free(host_ids);
  free(host_times);
  checkCudaErrors(cudaFree(device_ids));
  checkCudaErrors(cudaFree(device_times));
  checkCudaErrors(cudaFree(device_n_pmts_per_time_bin));
  free(host_max_number_of_pmts_in_time_bin);
  free(host_vertex_with_max_n_pmts);
  checkCudaErrors(cudaFree(device_max_number_of_pmts_in_time_bin));
  checkCudaErrors(cudaFree(device_vertex_with_max_n_pmts));

  return;
}

void free_global_memories(){

  //unbind texture reference to free resource 
  checkCudaErrors(cudaUnbindTexture(tex_times_of_flight));

  free(PMT_x);
  free(PMT_y);
  free(PMT_z);
  free(vertex_x);
  free(vertex_y);
  free(vertex_z);
  checkCudaErrors(cudaFree(device_times_of_flight));
  free(host_times_of_flight);

  return;
}

void copy_candidates_from_device_to_host(){

  checkCudaErrors(cudaMemcpy(host_max_number_of_pmts_in_time_bin,
			     device_max_number_of_pmts_in_time_bin,
			     n_time_bins*sizeof(unsigned int),
			     cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(host_vertex_with_max_n_pmts,
			     device_vertex_with_max_n_pmts,
			     n_time_bins*sizeof(unsigned int),
			     cudaMemcpyDeviceToHost));


}


void choose_candidates_above_threshold(){

  candidate_trigger_pair_vertex_time.clear();
  candidate_trigger_npmts_in_time_bin.clear();

  unsigned int the_threshold;

  for(unsigned int time_bin = 0; time_bin<n_time_bins - 1; time_bin++){ // loop over time bins
    // n_time_bins - 1 as we are checking the i and i+1 at the same time
    
    if(host_vertex_with_max_n_pmts[time_bin] < n_water_like_test_vertices )
      the_threshold = water_like_threshold_number_of_pmts;
    else
      the_threshold = wall_like_threshold_number_of_pmts;

    if(host_max_number_of_pmts_in_time_bin[time_bin] > the_threshold) {

      if( use_verbose )
	printf(" time %f vertex (%f, %f, %f) npmts %d \n", (time_bin + 2)*time_step_size - time_offset, vertex_x[host_vertex_with_max_n_pmts[time_bin]], vertex_y[host_vertex_with_max_n_pmts[time_bin]], vertex_z[host_vertex_with_max_n_pmts[time_bin]], host_max_number_of_pmts_in_time_bin[time_bin]);

      candidate_trigger_pair_vertex_time.push_back(std::make_pair(host_vertex_with_max_n_pmts[time_bin],time_bin+2));
      candidate_trigger_npmts_in_time_bin.push_back(host_max_number_of_pmts_in_time_bin[time_bin]);
    }

  }

  if( use_verbose )
    printf(" n candidates: %d \n", candidate_trigger_pair_vertex_time.size());
}





float read_value_from_file(std::string paramname, std::string filename){

  FILE * pFile = fopen (filename.c_str(),"r");

  char name[256];
  float value;

  while( EOF != fscanf(pFile, "%s %e", name, &value) ){
    if( paramname.compare(name) == 0 ){
      fclose(pFile);
      return value;
    }
  }

  printf(" warning: could not find parameter %s in file %s \n", paramname.c_str(), filename.c_str());

  fclose(pFile);
  exit(0);

  return 0.;

}

void read_user_parameters(std::string parameter_file){

  speed_light_water = 29.9792/1.3330; // speed of light in water, cm/ns
  //double speed_light_water = 22.490023;

  double dark_rate = read_value_from_file("dark_rate", parameter_file); // Hz
  distance_between_vertices = read_value_from_file("distance_between_vertices", parameter_file); // cm
  wall_like_distance = read_value_from_file("wall_like_distance", parameter_file); // units of distance between vertices
  time_step_size = (unsigned int)(sqrt(3.)*distance_between_vertices/(4.*speed_light_water)); // ns
  int extra_threshold = (int)(dark_rate*n_PMTs*2.*time_step_size*1.e-9); // to account for dark current occupancy
  water_like_threshold_number_of_pmts = read_value_from_file("water_like_threshold_number_of_pmts", parameter_file) + extra_threshold;
  wall_like_threshold_number_of_pmts = read_value_from_file("wall_like_threshold_number_of_pmts", parameter_file) + extra_threshold;
  coalesce_time = read_value_from_file("coalesce_time", parameter_file); // ns
  trigger_gate_up = read_value_from_file("trigger_gate_up", parameter_file); // ns
  trigger_gate_down = read_value_from_file("trigger_gate_down", parameter_file); // ns


}


int gpu_daq_initialize(std::string the_pmts_file,  std::string the_detector_file, std::string parameter_file){

  int argc = 0;
  const char* n_argv[] = {};
  const char **argv = n_argv;

  /////////////////////
  // initialise card //
  /////////////////////
  findCudaDevice(argc, argv);

  // initialise CUDA timing
  use_timing = true;
  if( use_timing ){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }
  cudaEventCreate(&total_start);
  cudaEventCreate(&total_stop);
  elapsed_parameters = 0; elapsed_pmts = 0; elapsed_detector = 0; elapsed_vertices = 0;
  elapsed_threads = 0; elapsed_tof = 0; elapsed_memory_tofs_dev = 0; elapsed_memory_candidates_host = 0; elapsed_tofs_copy_dev = 0;
  elapsed_input = 0; elapsed_memory_dev = 0; elapsed_copy_dev = 0; elapsed_kernel = 0; 
  elapsed_threads_candidates = 0; elapsed_candidates_memory_dev = 0; elapsed_candidates_kernel = 0;
  elapsed_candidates_copy_host = 0; choose_candidates = 0; elapsed_coalesce = 0; elapsed_gates = 0; elapsed_free = 0; elapsed_total = 0;
  elapsed_tofs_free = 0; elapsed_reset = 0;
  use_verbose = true;


  ////////////////////
  // inspect device //
  ////////////////////
  // set: max_n_threads_per_block, max_n_blocks
  print_gpu_properties();



  ////////////////
  // read PMTs  //
  ////////////////
  // set: n_PMTs, PMT_x, PMT_y, PMT_z
  if( use_timing )
    start_c_clock();
  detector_file = the_detector_file;
  pmts_file = the_pmts_file;
  if( !read_the_pmts() ) return 0;
  if( use_timing )
    elapsed_pmts = stop_c_clock();


  ///////////////////////
  // define parameters //
  ///////////////////////
  if( use_timing )
    start_c_clock();
  read_user_parameters(parameter_file);
  if( use_verbose ){
    printf(" --- user parameters \n");
    printf(" distance between test vertices = %f cm \n", distance_between_vertices);
    printf(" time step size = %d ns \n", time_step_size);
    printf(" water_like_threshold_number_of_pmts = %d \n", water_like_threshold_number_of_pmts);
    printf(" coalesce_time = %f ns \n", coalesce_time);
    printf(" trigger_gate_up = %f ns \n", trigger_gate_up);
    printf(" trigger_gate_down = %f ns \n", trigger_gate_down);
  }
  if( use_timing )
    elapsed_parameters = stop_c_clock();




  /////////////////////
  // read detector ////
  /////////////////////
  // set: detector_height, detector_radius, pmt_radius
  if( use_timing )
    start_c_clock();
  if( !read_the_detector() ) return 0;
  if( use_timing )
    elapsed_detector = stop_c_clock();




  ////////////////////////
  // make test vertices //
  ////////////////////////
  // set: n_test_vertices, n_water_like_test_vertices, vertex_x, vertex_y, vertex_z
  // use: detector_height, detector_radius
  if( use_timing )
    start_c_clock();
  make_test_vertices();
  if( use_timing )
    elapsed_vertices = stop_c_clock();



  //////////////////////////////
  // table of times_of_flight //
  //////////////////////////////
  // set: host_times_of_flight, time_offset
  // use: n_test_vertices, vertex_x, vertex_y, vertex_z, n_PMTs, PMT_x, PMT_y, PMT_z
  // malloc: host_times_of_flight
  if( use_timing )
    start_c_clock();
  make_table_of_tofs();
  if( use_timing )
    elapsed_tof = stop_c_clock();



  ////////////////////////////////////
  // allocate tofs memory on device //
  ////////////////////////////////////
  // use: n_test_vertices, n_PMTs
  // cudamalloc: device_times_of_flight
  if( use_timing )
    start_cuda_clock();
  allocate_tofs_memory_on_device();
  if( use_timing )
    elapsed_memory_tofs_dev = stop_cuda_clock();


  ////////////////////////////////
  // fill tofs memory on device //
  ////////////////////////////////
  // use: n_test_vertices, n_water_like_test_vertices, n_PMTs
  // memcpy: device_times_of_flight, constant_time_step_size, constant_n_test_vertices, constant_n_water_like_test_vertices, constant_n_PMTs
  // texture: tex_times_of_flight
  if( use_timing )
    start_cuda_clock();
  fill_tofs_memory_on_device();
  if( use_timing )
    elapsed_tofs_copy_dev = stop_cuda_clock();




  start_total_cuda_clock();

  return 1;

}


int gpu_daq_finalize(){


  elapsed_total += stop_total_cuda_clock();


  //////////////////////////////
  // deallocate global memory //
  //////////////////////////////
  if( use_timing )
    start_cuda_clock();
  if( use_verbose )
    printf(" --- deallocate tofs memory \n");
  free_global_memories();
  if( use_timing )
    elapsed_tofs_free = stop_cuda_clock();



  //////////////////
  // reset device //
  //////////////////
  // -- needed to flush the buffer which holds printf from each thread
  if( use_timing )
    start_cuda_clock();
  if( use_verbose )
    printf(" --- reset device \n");
  //  cudaDeviceReset();
  if( use_timing )
    elapsed_reset = stop_cuda_clock();



  //////////////////
  // print timing //
  //////////////////
  if( use_timing ){
    printf(" user parameters time : %f ms \n", elapsed_parameters);
    printf(" read pmts execution time : %f ms \n", elapsed_pmts);
    printf(" read detector execution time : %f ms \n", elapsed_detector);
    printf(" make test vertices execution time : %f ms \n", elapsed_vertices);
    printf(" setup threads candidates execution time : %f ms \n", elapsed_threads_candidates);
    printf(" make table of tofs execution time : %f ms \n", elapsed_tof);
    printf(" allocate tofs memory on device execution time : %f ms \n", elapsed_memory_tofs_dev);
    printf(" fill tofs memory on device execution time : %f ms \n", elapsed_tofs_copy_dev);
    printf(" deallocate tofs memory execution time : %f ms \n", elapsed_tofs_free);
    printf(" device reset execution time : %f ms \n", elapsed_reset);
    printf(" read input execution time : %f ms (%f) \n", elapsed_input, elapsed_input/elapsed_total);
    printf(" allocate candidates memory on host execution time : %f ms (%f) \n", elapsed_memory_candidates_host, elapsed_memory_candidates_host/elapsed_total);
    printf(" setup threads execution time : %f ms (%f) \n", elapsed_threads, elapsed_threads/elapsed_total);
    printf(" allocate memory on device execution time : %f ms (%f) \n", elapsed_memory_dev, elapsed_memory_dev/elapsed_total);
    printf(" fill memory on device execution time : %f ms (%f) \n", elapsed_copy_dev, elapsed_copy_dev/elapsed_total);
    printf(" correct kernel execution time : %f ms (%f) \n", elapsed_kernel, elapsed_kernel/elapsed_total);
    printf(" allocate candidates memory on device execution time : %f ms (%f) \n", elapsed_candidates_memory_dev, elapsed_candidates_memory_dev/elapsed_total);
    printf(" copy candidates to host execution time : %f ms (%f) \n", elapsed_candidates_copy_host, elapsed_candidates_copy_host/elapsed_total);
    printf(" choose candidates execution time : %f ms (%f) \n", choose_candidates, choose_candidates/elapsed_total);
    printf(" candidates kernel execution time : %f ms (%f) \n", elapsed_candidates_kernel, elapsed_candidates_kernel/elapsed_total);
    printf(" coalesce triggers execution time : %f ms (%f) \n", elapsed_coalesce, elapsed_coalesce/elapsed_total);
    printf(" separate triggers into gates execution time : %f ms (%f) \n", elapsed_gates, elapsed_gates/elapsed_total);
    printf(" deallocate memory execution time : %f ms (%f) \n", elapsed_free, elapsed_free/elapsed_total);
  }
  printf(" total execution time : %f ms \n", elapsed_total);

  return 1;

}
