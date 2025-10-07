#ifndef ACOTSP_H_
#define ACOTSP_H_

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

constexpr int MAX_N = 1024;

#ifndef DEBUG
constexpr bool debug = false;
#else
constexpr bool debug = true;
#endif

template <typename... Args> inline void die(Args &&...args)
{
    static std::ios_base::Init init;
    (std::cerr << ... << std::forward<Args>(args)) << std::endl;
    std::exit(EXIT_FAILURE);
}

class acotsp
{
  public:
    enum class Type { WORKER, QUEEN };
    enum class EdgeWeightType { EUC_2D, CEIL_2D, GEO };

    // Configuration parameters
    Type type;
    int num_iter;
    float alpha;
    float beta;
    float rho;
    size_t seed;
    std::string input_file;
    std::string output_file;

    // Problem data
    int dimension;
    EdgeWeightType edge_weight_type;
    std::vector<float> xs;
    std::vector<float> ys;

    acotsp(int argc, char *argv[])
    {
        if (argc != 1 + 8)
            die("Not enough arguments\n");

        parse_arguments(argv);
        read_input_file();
    }

  private:
    void parse_arguments(char *argv[])
    {
        input_file = argv[1];
        output_file = argv[2];

        std::string type_str = argv[3];
        if (type_str == "WORKER")
            type = Type::WORKER;
        else if (type_str == "QUEEN")
            type = Type::QUEEN;
        else
            die("Incorrect type");

        num_iter = std::stoi(argv[4]);
        alpha = std::stof(argv[5]);
        beta = std::stof(argv[6]);
        rho = std::stof(argv[7]);

        if (!(0 <= rho && rho <= 1))
            die("Incorrect evaporate value");

        seed = std::stoull(argv[8]);
    }

    void read_input_file()
    {
        std::ifstream file(input_file);
        if (!file.is_open())
            die("Error opening file: ", input_file);

        std::string line;

        // Read and discard NAME, COMMENT, TYPE lines
        for (int i = 0; i < 3; ++i) {
            if (!getline(file, line))
                die("Error reading header line: ", line);
        }

        // Read DIMENSION
        if (!getline(file, line))
            die("Error reading DIMENSION: ", line);

        dimension = std::stoi(line.substr(line.find(":") + 1));
        if (!(1 <= dimension && dimension <= MAX_N))
            die("Incorrect dimension");

        // Read EDGE_WEIGHT_TYPE
        if (!getline(file, line))
            die("Error reading EDGE_WEIGHT_TYPE: ", line);

        std::string edge_type = line.substr(line.find(":") + 1);
        edge_type.erase(0, edge_type.find_first_not_of(" \t"));
        edge_type.erase(edge_type.find_last_not_of(" \t") + 1);
        if (edge_type == "EUC_2D")
            edge_weight_type = EdgeWeightType::EUC_2D;
        else if (edge_type == "CEIL_2D")
            edge_weight_type = EdgeWeightType::CEIL_2D;
        else if (edge_type == "GEO")
            edge_weight_type = EdgeWeightType::GEO;
        else
            die("Unknown EDGE_WEIGHT_TYPE: ", edge_type);

        // Verify NODE_COORD_SECTION
        if (!getline(file, line) || line != "NODE_COORD_SECTION")
            die("Error reading NODE_COORD_SECTION: ", line);

        // Read coordinates
        xs.reserve(dimension);
        ys.reserve(dimension);
        for (int i = 0; i < dimension; ++i) {
            if (!getline(file, line))
                die("Error reading coordinates: ", line);

            std::istringstream iss(line);
            int index;
            float x, y;
            iss >> index >> x >> y;
            xs.push_back(x);
            ys.push_back(y);
        }

        // Verify EOF
        if (!getline(file, line) || line != "EOF")
            die("Error reading EOF: ", line);
    }

  public:
    static float nint(float x) { return (float)((int)(x + 0.5)); }

    float distance(int i, int j) const
    {
        switch (edge_weight_type) {
        case EdgeWeightType::EUC_2D: {
            const float dx = xs[i] - xs[j];
            const float dy = ys[i] - ys[j];
            return nint(std::sqrt(dx * dx + dy * dy));
        }

        case EdgeWeightType::CEIL_2D: {
            const float dx = xs[i] - xs[j];
            const float dy = ys[i] - ys[j];
            return std::ceil(std::sqrt(dx * dx + dy * dy));
        }

        case EdgeWeightType::GEO: {
            constexpr float RRR = 6378.388f;
            const float lat1 = deg2radians(xs[i]);
            const float lon1 = deg2radians(ys[i]);
            const float lat2 = deg2radians(xs[j]);
            const float lon2 = deg2radians(ys[j]);

            const float q1 = std::cos(lon1 - lon2);
            const float q2 = std::cos(lat1 - lat2);
            const float q3 = std::cos(lat1 + lat2);

            return std::round(
                RRR * std::acos(0.5f * ((1.0f + q1) * q2 - (1.0f - q1) * q3)) +
                1.0f);
        }

        default: die("Unknown EDGE_WEIGHT_TYPE"); return 0; // unreachable
        }
    }

    float deg2radians(float degree) const
    {
        int deg = static_cast<int>(degree);
        float min = degree - deg;
        return M_PI * (deg + 5.0f * min / 3.0f) / 180.0f;
    }
};

#endif /* ACOTSP_H_ */
