#include <algorithm>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "acotsp.hpp"
#include "utils.cu"

struct Arg {
    int N;
    curandState *state;
    float alpha;
    float beta;
    float rho;
    float *tau;
    float *eta;
    float *choice_info;
    int *distances;
    int max_distance;
    int *tours;
    int *tour_lengths;
};

// device {{{
template <typename T> __device__ void swap(T *a, T *b)
{
    T t = *a;
    *a = *b;
    *b = t;
}

template <typename T> __device__ void reduce(volatile T *t, int tx)
{
#pragma unroll
    for (unsigned s = MAX_N / 2; s > 32; s >>= 1) {
        if (tx < s)
            t[tx] += t[tx + s];
        __syncthreads();
    }

    if (tx < 32) {
        T val = t[tx] + t[tx + 32];
#pragma unroll
        for (unsigned s = 16; s > 0; s >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, s);
        t[tx] = val;
    }
}
// }}}

__global__ void setup_curand(int N, curandState *state, size_t seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, i, 0, &state[i]);
}

__global__ void setup_ants(Arg *arg)
{
    const int N = arg->N;
    const float beta = arg->beta;
    float *tau = arg->tau;
    float *eta = arg->eta;
    const int *distances = arg->distances;
    const int max_distance = arg->max_distance;

    int i = threadIdx.x;
    int j = blockIdx.x;
    if (i >= N || j >= N)
        return;

    float d = distances[i * N + j];
    d = d ? 1 / d : 1;
    tau[i * N + j] = 1;
    eta[i * N + j] = __powf(d * max_distance, beta);
}

__global__ void update_choice_info(Arg *arg)
{
    const int N = arg->N;
    const float alpha = arg->alpha;
    const float *tau = arg->tau;
    const float *eta = arg->eta;
    float *choice_info = arg->choice_info;

    int i = threadIdx.x;
    int j = blockIdx.x;
    if (i >= N || j >= N)
        return;

    choice_info[i * N + j] = __powf(tau[i * N + j], alpha) * eta[i * N + j];
}

__global__ void worker_tour_construction(Arg *arg)
{
    const int N = arg->N;
    curandState *state = arg->state;
    const float *choice_info = arg->choice_info;
    int *tours = arg->tours;

    int k = blockIdx.x;
    if (k >= N)
        return;

    __shared__ int tour[MAX_N];
    __shared__ float choice[MAX_N];

    for (int j = 0; j < N; j++)
        tour[j] = j;

    int step = 0;
    int i = 0; /* NOTE: no need to choose at random */
    while (++step < N) {
        for (int j = step; j < N; j++)
            choice[j] = choice_info[i * N + tour[j]];

        float sum = 0;
        for (int j = step; j < N; j++)
            sum += choice[j];

        float p = curand_uniform(&state[k]) * sum;
        for (int j = step; j < N; j++) {
            float rp = choice[j];

            if (p <= rp) {
                swap(&tour[step], &tour[j]);
                break;
            }
            p -= rp;
        }

        i = tour[step];
    }

    for (int j = 0; j < N; j++)
        tours[k * MAX_N + j] = tour[j];
}

__global__ void queen_tour_construction(Arg *arg)
{
#define NUM_BANKS     16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n)                                                \
    (((n) >> NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)))
#define INDEX(i) ((i) + CONFLICT_FREE_OFFSET(i))

    const int N = arg->N;
    curandState_t *state = arg->state;
    const float *choice_info = arg->choice_info;
    int *tours = arg->tours;

    int tx = threadIdx.x;
    int ant = blockIdx.x;

    if (ant >= N)
        return;

    __shared__ int tour[MAX_N], candidate;
    __shared__ float prob[MAX_N + MAX_N / NUM_BANKS], sum, p;

    tour[tx] = !tx ? 0 : MAX_N;
    __syncthreads();

    int step = 0;
    int i = 0;
    bool tabu = !tx || tx >= N;
    while (++step < N) {
        prob[INDEX(tx)] = tabu ? 0 : choice_info[i * N + tx];
        if (!tx)
            p = curand_uniform(&state[ant]);
        __syncthreads();

        if (tx == MAX_N - 1)
            sum = prob[INDEX(tx)];

        /* based on the paper:
         * https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
         */
        int offset = 1;
#pragma unroll
        for (int d = MAX_N >> 1; d > 0; d >>= 1) {
            __syncthreads();
            if (tx < d) {
                int ai = offset * (2 * tx + 1) - 1;
                int bi = offset * (2 * tx + 2) - 1;
                prob[INDEX(bi)] += prob[INDEX(ai)];
            }
            offset <<= 1;
        }

        if (tx == 0)
            prob[INDEX(MAX_N - 1)] = 0;

#pragma unroll
        for (int d = 1; d < MAX_N; d <<= 1) {
            offset >>= 1;
            __syncthreads();
            if (tx < d) {
                int ai = offset * (2 * tx + 1) - 1;
                int bi = offset * (2 * tx + 2) - 1;
                float t = prob[INDEX(ai)];
                prob[INDEX(ai)] = prob[INDEX(bi)];
                prob[INDEX(bi)] += t;
            }
        }
        __syncthreads();

        if (tx == MAX_N - 1)
            sum += prob[INDEX(tx)];
        __syncthreads();

        prob[INDEX(tx)] /= sum;
        __syncthreads();

        constexpr float eps = 1.0f / (MAX_N * MAX_N); /* NOTE: floats... */
        float lp = prob[INDEX(tx)] - eps;
        float rp = (tx == MAX_N - 1 ? 1.0f : prob[INDEX(tx + 1)]) + eps;
        if (!tabu && lp < p && p <= rp)
            candidate = tx; /* nondeterministic choice */
        __syncthreads();

        i = candidate;

        if (tx == i) {
            tabu = true;
            tour[step] = tx;
        }
    }
    __syncthreads();

    tours[ant * MAX_N + tx] = tour[tx];
}

__global__ void pheromone_update_stage1(Arg *arg)
{
    const int N = arg->N;
    const float rho = arg->rho;
    float *tau = arg->tau;

    int ant = blockIdx.x;
    int tx = threadIdx.x;
    if (ant >= N || tx >= N)
        return;

    float f = tau[ant * N + tx];
    constexpr float bound = 1.0f / (float)MAX_N;
    f *= rho;
    f = f < bound ? bound : f; /* NOTE: in order to prevent zero probabilities,
                                  we enforce a lower bound */
    tau[ant * N + tx] = f;
}

__global__ void pheromone_update_stage2(Arg *arg)
{
    const int N = arg->N;
    float *tau = arg->tau;
    const int *distances = arg->distances;
    const int *tours = arg->tours;
    int *tour_lengths = arg->tour_lengths;

    int ant = blockIdx.x;
    int tx = threadIdx.x;
    if (ant >= N)
        return;

    __shared__ int tour[MAX_N];
    __shared__ int C[MAX_N];
    __shared__ float dist;

    tour[tx] = tours[ant * MAX_N + tx];
    __syncthreads();

    int i = tour[tx];
    int j = tx == N - 1 ? tour[0] : tour[tx + 1];

    C[tx] = tx < N ? distances[i * N + j] : 0;
    __syncthreads();

    reduce(C, tx); /* sum of elements placed in C[0] */

    if (tx == 0) {
        tour_lengths[ant] = C[0];
        dist = (float)arg->max_distance /
               (float)C[0]; /* NOTE: multiplying by max distance fixes the issue
                               of small floats */
    } else if (tx >= N)
        return;
    __syncthreads();

    // TODO: scatter to gather
    // [(i0,j0),(i1,j1),...]
    // [(0,j0),(1,j1),...]
    // transpose
    // [(0,j0),(0,j1),...]
    // ...
    // [(n-1,j0),(n-1,j1)...]
    // sort rows
    // [(0,1),(0,2),...]
    // enum (get idx of first j in a row)
    // sum segments [j_first..(j+1)_first-1]
    // add sums to tau[row * N + j]

    // NOTE: orders rows
    // tour[i] = j;
    // __syncthreads();
    // i = tx;
    // j = tour[tx];

    atomicAdd(&tau[i * N + j], dist);
    atomicAdd(&tau[j * N + i], dist);
}

__global__ void find_best_tour(Arg *arg, int *best_ant)
{
    int N = arg->N;
    int *tour_lengths = arg->tour_lengths;

    unsigned k = threadIdx.x;

    __shared__ int C[MAX_N];
    __shared__ int ant[MAX_N];

    if (k < N) {
        C[k] = tour_lengths[k];
        ant[k] = k;
    } else {
        C[k] = tour_lengths[0];
        ant[k] = 0;
    }
    __syncthreads();

#pragma unroll
    for (unsigned s = MAX_N / 2; s > 32; s >>= 1) {
        if (k < s) {
            if (C[k] > C[k + s]) {
                C[k] = C[k + s];
                ant[k] = ant[k + s];
            }
        }
        __syncthreads();
    }

    if (k < 32) {
#pragma unroll
        for (unsigned s = 32; s > 0; s >>= 1) {
            if (C[k] > C[k + s]) {
                C[k] = C[k + s];
                ant[k] = ant[k + s];
            }
        }

        if (k == 0)
            *best_ant = ant[0];
    }
}

struct GraphResources {
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    Arg *arg;
    int *best_ant;
};

template <acotsp::Type type>
void create_graph(acotsp tsp, GraphResources &resources)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraphCreate(&resources.graph, 0);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    update_choice_info<<<MAX_N, MAX_N, 0, stream>>>(resources.arg);
    if constexpr (type == acotsp::Type::WORKER) {
        worker_tour_construction<<<MAX_N, 1, 0, stream>>>(resources.arg);
    } else {
        queen_tour_construction<<<MAX_N, MAX_N, 0, stream>>>(resources.arg);
    }
    pheromone_update_stage1<<<MAX_N, MAX_N, 0, stream>>>(resources.arg);
    pheromone_update_stage2<<<MAX_N, MAX_N, 0, stream>>>(resources.arg);
    find_best_tour<<<1, MAX_N, 0, stream>>>(resources.arg, resources.best_ant);

    cudaStreamEndCapture(stream, &resources.graph);
    cudaGraphInstantiate(&resources.instance, resources.graph, NULL, NULL, 0);

    cudaStreamDestroy(stream);
}

template <acotsp::Type type, bool use_graph> void run(acotsp tsp)
{
    int N = tsp.dimension;

    int distances_h[N * N];
    int tour[N];
    int ant;
    int C, best_C = INT_MAX;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            distances_h[i * N + j] = tsp.distance(i, j);
    }
    int max_distance = *std::max_element(distances_h, distances_h + N * N);

    cuda_array<curandState> state_d(N);
    cuda_array<float> tau_d(N * N);
    cuda_array<float> eta_d(N * N);
    cuda_array<float> choice_info_d(N * N);
    cuda_array<int> distances_d(distances_h, N * N);
    cuda_array<int> tours_d(MAX_N * MAX_N);
    cuda_array<int> tour_lengths_d(MAX_N);
    cuda_array<int> ant_d(1);

    setup_curand<<<1, MAX_N>>>(N, state_d, tsp.seed);

    Arg arg_h{.N = N,
              .state = state_d,
              .alpha = tsp.alpha,
              .beta = tsp.beta,
              .rho = 1 - tsp.rho,
              .tau = tau_d,
              .eta = eta_d,
              .choice_info = choice_info_d,
              .distances = distances_d,
              .max_distance = max_distance,
              .tours = tours_d,
              .tour_lengths = tour_lengths_d};
    cuda_array<Arg> arg_d(&arg_h, 1);

    setup_ants<<<MAX_N, MAX_N>>>(arg_d);

    cuda_timer all_timer;
    cuda_timer update_choice_info_timer;
    cuda_timer tour_construction_timer;
    cuda_timer pheromone_update_timer;
    cuda_timer find_best_tour_timer;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    GraphResources graph_resources = {
        .arg = arg_d,
        .best_ant = ant_d,
    };
    create_graph<type>(tsp, graph_resources);

    for (int iter = 0; iter < tsp.num_iter; iter++) {
        all_timer.start();

        if constexpr (use_graph) {
            cudaGraphLaunch(graph_resources.instance, stream);
            cudaStreamSynchronize(stream);
        } else {
            update_choice_info_timer.start();
            update_choice_info<<<MAX_N, MAX_N>>>(arg_d);
            update_choice_info_timer.stop();

            tour_construction_timer.start();
            if constexpr (type == acotsp::Type::WORKER) {
                worker_tour_construction<<<MAX_N, 1>>>(arg_d);
            } else {
                queen_tour_construction<<<MAX_N, MAX_N>>>(arg_d);
            }
            tour_construction_timer.stop();

            pheromone_update_timer.start();
            pheromone_update_stage1<<<MAX_N, MAX_N>>>(arg_d);
            pheromone_update_stage2<<<MAX_N, MAX_N>>>(arg_d);
            pheromone_update_timer.stop();

            find_best_tour_timer.start();
            find_best_tour<<<1, MAX_N>>>(arg_d, ant_d);
            find_best_tour_timer.stop();
        }

        all_timer.stop();

        cudaMemcpy(&ant, ant_d, sizeof(ant), cudaMemcpyDeviceToHost);
        cudaMemcpy(&C, &tour_lengths_d[ant], sizeof(C), cudaMemcpyDeviceToHost);

        if (C < best_C) {
            best_C = C;
            cudaMemcpy(tour, &tours_d[ant * MAX_N], sizeof(tour),
                       cudaMemcpyDeviceToHost);
        }

        if (!debug && iter < tsp.num_iter - 1)
            continue;

        if (debug) {
            fprintf(stderr, "[%*d] %d\n",
                    (int)std::ceil(std::log10(tsp.num_iter + 1)), iter + 1, C);
        }
    }

    cudaGraphDestroy(graph_resources.graph);
    cudaGraphExecDestroy(graph_resources.instance);
    cudaStreamDestroy(stream);

    if constexpr (type == acotsp::Type::WORKER) {
        all_timer.print(tsp.input_file + " [WORKER]");
    } else {
        all_timer.print(tsp.input_file + " [QUEEN]");
    }

    if constexpr (!use_graph) {
        update_choice_info_timer.print("update_choice_info");
        tour_construction_timer.print("tour_construction");
        pheromone_update_timer.print("pheromone_update");
        find_best_tour_timer.print("find_best_tour");
    }

    FILE *fp = fopen(tsp.output_file.c_str(), "w");
    if (!fp)
        die("cannot open output file");
    fprintf(fp, "%d\n", best_C);
    for (int i = 0; i < N; i++)
        fprintf(fp, "%d%c", tour[i] + 1, i == N - 1 ? '\n' : ' ');
    fclose(fp);
}

int main(int argc, char *argv[])
{
    acotsp args(argc, argv);

    switch (args.type) {
    case acotsp::Type::WORKER: run<acotsp::Type::WORKER, true>(args); break;
    case acotsp::Type::QUEEN:  run<acotsp::Type::QUEEN, true>(args); break;
    }

    return 0;
}
