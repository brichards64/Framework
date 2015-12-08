
#ifndef LIBRARY_DAQ_H
#define LIBRARY_DAQ_H


int CUDAFunction(std::vector<int> PMTids, std::vector<int> times);

int gpu_daq_initialize(std::string pmts_file,  std::string detector_file, std::string parameter_file);
int gpu_daq_finalize();

bool read_input(std::vector<int> PMTids, std::vector<int> times, int * max_time);
void print_parameters();
void print_parameters_2d();
void print_input();
void print_times_of_flight();
void print_pmts();
bool read_detector();
bool read_the_pmts();
bool read_the_detector();
void make_test_vertices();
bool setup_threads_for_tof();
bool setup_threads_for_tof_biparallel();
bool setup_threads_for_tof_2d();
bool setup_threads_to_find_candidates();
bool read_the_input(std::vector<int> PMTids, std::vector<int> times);
void allocate_tofs_memory_on_device();
void allocate_correct_memory_on_device();
void allocate_candidates_memory_on_host();
void allocate_candidates_memory_on_device();
void make_table_of_tofs();
void fill_correct_memory_on_device();
void fill_tofs_memory_on_device();
void coalesce_triggers();
void separate_triggers_into_gates();
float timedifference_msec(struct timeval t0, struct timeval t1);
void start_c_clock();
double stop_c_clock();
void start_cuda_clock();
double stop_cuda_clock();
void start_total_cuda_clock();
double stop_total_cuda_clock();
unsigned int get_distance_index(unsigned int pmt_id, unsigned int vertex_block);
unsigned int get_time_index(unsigned int hit_index, unsigned int vertex_block);
void print_gpu_properties();
unsigned int read_number_of_pmts();
bool read_pmts();
void free_event_memories();
void free_global_memories();
void copy_candidates_from_device_to_host();
void choose_candidates_above_threshold();
float read_value_from_file(std::string paramname, std::string filename);
void read_user_parameters(std::string parameter_file);


#endif

