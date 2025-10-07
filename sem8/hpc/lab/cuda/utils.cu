#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

// utils {{{
template <typename T> class cuda_array
{
  public:
    cuda_array(size_t size) : _size(size)
    {
        cudaMalloc(&_data, size * sizeof(T));
    }

    cuda_array(const T *h_data, size_t size) : _size(size)
    {
        cudaMalloc(&_data, size * sizeof(T));
        cudaMemcpy(_data, h_data, size * sizeof(T), cudaMemcpyHostToDevice);
    }

    ~cuda_array() { cudaFree(_data); }

    operator T *() { return _data; }

  private:
    T *_data;
    size_t _size;
};

class cuda_timer
{
  public:
    cuda_timer()
    {
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
    }

    void start() { cudaEventRecord(_start, 0); }

    float stop()
    {
        cudaEventRecord(_stop, 0);
        cudaEventSynchronize(_stop);

        float ms;
        cudaEventElapsedTime(&ms, _start, _stop);

        _times.emplace_back(ms);

        return ms;
    }

    float mean() const
    {
        return std::accumulate(_times.begin(), _times.end(), 0.0) /
               _times.size();
    }

    float std() const
    {
        float mean = this->mean();
        float var = 0;
        for (const auto &ms : _times)
            var += (ms - mean) * (ms - mean);
        var /= _times.size();

        return sqrt(var);
    }

    std::vector<float> get_times() { return _times; }
    void clear() { _times.clear(); }

    void print(const std::string &name)
    {
        float mean = this->mean();
        float std = this->std();
        std::cout << name << " | " << format_float(mean) << "+/-"
                  << format_float(std) << " | " << " -- "
                  << " |" << std::endl;
    }

    void print(const std::string &name, const cuda_timer &reference)
    {
        float mean = this->mean();
        float std = this->std();
        float mean_ref = reference.mean();
        float std_ref = reference.std();
        std::cout << name << " | " << format_float(mean) << "+/-"
                  << format_float(std) << " | " << format_float(mean_ref / mean)
                  << "+/-"
                  << format_float(
                         sqrt(pow(mean * std_ref, 2) + pow(std * mean, 2)) /
                         pow(mean, 2))
                  << " | " << std::endl;
    }

    ~cuda_timer()
    {
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }

  private:
    cudaEvent_t _start;
    cudaEvent_t _stop;
    std::vector<float> _times;

    std::string format_float(float ms) const
    {
        std::string str = "";
        str += std::to_string(int(ms)) + ".";
        ms -= int(ms);
        if (ms == 0)
            return str + "0";
        while ((int)ms == 0)
            ms *= 10;
        return str + std::to_string(static_cast<int>(std::round(ms * 10)));
    }
};

template <typename... Args> struct cuda_bench {
    std::string name;
    void (*f)(Args...);
    int runs;
    std::tuple<Args...> args;

    cuda_bench(std::string name, void (*f)(Args...), int runs = 5)
        : name(name), f(f), runs(runs)
    {
    }

    void set_args(Args... args) { this->args = std::tuple<Args...>(args...); }

    template <typename... Bench> void bench(Bench... args)
    {
        cuda_timer reference;
        for (int i = 0; i < runs; i++) {
            reference.start();
            std::apply(f, this->args);
            reference.stop();
        }
        reference.print(name);

        for (auto &b : {args...}) {
            cuda_timer timer;
            for (int i = 0; i < b.runs; i++) {
                timer.start();
                std::apply(b.f, b.args);
                timer.stop();
            }
            timer.print(b.name, reference);
        }
    }
};
// }}}
