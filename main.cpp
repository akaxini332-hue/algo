#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <set>
#include <map>
#include <fstream>
#include <iomanip>


class HashFuncGen {
public:
    static uint32_t hash(const std::string& key, uint32_t seed = 42) {
        const char* data = key.data();
        const int len = key.length();
        const int nblocks = len / 4;
        uint32_t h1 = seed;
        const uint32_t c1 = 0xcc9e2d51;
        const uint32_t c2 = 0x1b873593;
        const uint32_t* blocks = (const uint32_t*)(data + nblocks * 4);
        for (int i = -nblocks; i; i++) {
            uint32_t k1 = blocks[i];
            k1 *= c1; k1 = (k1 << 15) | (k1 >> 17); k1 *= c2;
            h1 ^= k1; h1 = (h1 << 13) | (h1 >> 19); h1 = h1 * 5 + 0xe6546b64;
        }
        const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
        uint32_t k1 = 0;
        switch (len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1; k1 = (k1 << 15) | (k1 >> 17); k1 *= c2; h1 ^= k1;
        }
        h1 ^= len; h1 ^= h1 >> 16; h1 *= 0x85ebca6b; h1 ^= h1 >> 13; h1 *= 0xc2b2ae35; h1 ^= h1 >> 16;
        return h1;
    }
};

class RandomStreamGen {
private:
    std::mt19937 rng;
    std::string charset;
    size_t total_items;
    size_t current_pos;

    std::string generate_random_string(size_t max_length) {
        std::uniform_int_distribution<size_t> len_dist(5, max_length);
        std::uniform_int_distribution<size_t> char_dist(0, charset.size() - 1);
        size_t length = len_dist(rng);
        std::string result; result.reserve(length);
        for (size_t i = 0; i < length; ++i) result += charset[char_dist(rng)];
        return result;
    }

public:
    RandomStreamGen(size_t n_items, unsigned int seed) : total_items(n_items), current_pos(0) {
        rng.seed(seed);
        for (char c = 'a'; c <= 'z'; ++c) charset += c;
        for (char c = 'A'; c <= 'Z'; ++c) charset += c;
        for (char c = '0'; c <= '9'; ++c) charset += c;
        charset += '-';
    }
    bool has_next() const { return current_pos < total_items; }
    
    std::vector<std::string> next_batch(double fraction) {
        size_t batch_size = static_cast<size_t>(total_items * fraction);
        if (batch_size == 0) batch_size = 1;
        if (current_pos + batch_size > total_items) batch_size = total_items - current_pos;
        std::vector<std::string> batch; batch.reserve(batch_size);
        for (size_t i = 0; i < batch_size; ++i) batch.push_back(generate_random_string(30));
        current_pos += batch_size;
        return batch;
    }
};



class HyperLogLog {
private:
    int b;
    size_t m;
    double alphaMM;
    std::vector<uint8_t> registers;

    void calculate_alphaMM() {
        double alpha;
        switch (m) {
            case 16: alpha = 0.673; break;
            case 32: alpha = 0.697; break;
            case 64: alpha = 0.709; break;
            default: alpha = 0.7213 / (1.0 + 1.079 / m); break;
        }
        alphaMM = alpha * m * m;
    }

    uint8_t get_rho(uint32_t w) {
        if (w == 0) return 32 - b + 1;
        uint8_t rho = 1;
        while ((w & 0x80000000) == 0) { w <<= 1; rho++; }
        return rho;
    }

public:
    HyperLogLog(int b_bits) : b(b_bits) {
        if (b < 4) b = 4; if (b > 16) b = 16;
        m = 1 << b;
        registers.assign(m, 0);
        calculate_alphaMM();
    }

    void add(const std::string& item) {
        uint32_t x = HashFuncGen::hash(item);
        uint32_t j = x >> (32 - b); 
        uint32_t w = x << b;        
        uint8_t rho = get_rho(w);
        if (rho > registers[j]) registers[j] = rho;
    }

    double estimate() const {
        double sum_inv = 0.0;
        int zeros_count = 0;
        for (uint8_t val : registers) {
            sum_inv += std::pow(2.0, -val);
            if (val == 0) zeros_count++;
        }
        double E = alphaMM / sum_inv;

       
        if (E <= 2.5 * m) {
            if (zeros_count > 0) {
                E = m * std::log(static_cast<double>(m) / zeros_count);
            }
        } 
      
        else if (E > 4294967296.0 / 30.0) {
            E = -4294967296.0 * std::log(1.0 - E / 4294967296.0);
        }
        return E;
    }
    
    
    double get_occupancy() const {
        int filled = 0;
        for (uint8_t r : registers) if (r > 0) filled++;
        return (double)filled / m * 100.0;
    }
};


int calibrate_and_choose_B() {
    std::cout << "\n выбор B\n";
    std::cout << "Проводим тестовые запуски для оценки точности и распределения\n";
    
    std::vector<int> b_candidates = {10, 12, 14, 16};
    
    std::cout << std::setw(5) << "B" << " | " << std::setw(15) << "Регистры (m)" 
              << " | " << std::setw(15) << "Заполн.(%)" << " | " << "Ошибка(%)" << "\n";
    std::cout << std::string(65, '-') << "\n";

    int best_b = 12;
    double min_error = 100.0;

    for(int b : b_candidates) {
        
        RandomStreamGen gen(50000, 111); 
        HyperLogLog hll(b);
        std::set<std::string> exact;
        
   
        while(gen.has_next()) {
            auto batch = gen.next_batch(1.0);
            for(const auto& s : batch) {
                hll.add(s);
                exact.insert(s);
            }
        }
        
        double est = hll.estimate();
        double err = std::abs(est - (double)exact.size()) / exact.size() * 100.0;
        double occ = hll.get_occupancy();
        
        std::cout << std::setw(5) << b << " | " 
                  << std::setw(15) << (1<<b) << " | " 
                  << std::setw(15) << std::fixed << std::setprecision(2) << occ << " | " 
                  << std::setprecision(4) << err << "\n";
                  
      
        if (err < min_error) {
            min_error = err;
            best_b = b;
        }
    }
    std::cout << ">>> Выбрано B = " << best_b << " (Обеспечивает минимальную ошибку)\n";
    return best_b;
}

int main() {
    
    setlocale(LC_ALL, "");


    int B = calibrate_and_choose_B();

    
    const int NUM_RUNS = 10;            
    const size_t STREAM_SIZE = 100000;  
    const double STEP = 0.05;          


    struct RunData {
        size_t true_val;
        double est_val;
    };
    
    std::map<int, std::vector<RunData>> collected_data;
    std::vector<double> fractions; 

    std::cout << "\nГенерация потоков и сбор статистики\n";
    std::cout << "Запуск " << NUM_RUNS << " независимых потоков по " << STREAM_SIZE << " элементов\n";

    for (int run = 0; run < NUM_RUNS; ++run) {
        RandomStreamGen gen(STREAM_SIZE, 500 + run); 
        HyperLogLog hll(B);
        std::set<std::string> exact_counter;

        int step_idx = 0;
        double current_frac = 0.0;
        
        if (run % 2 == 0) std::cout << "Обработка потока №" << (run + 1) << "\n";

        while(gen.has_next()) {
            auto batch = gen.next_batch(STEP);
            current_frac += STEP;
            step_idx++;

            if (run == 0) fractions.push_back(current_frac);

            for(const auto& s : batch) {
                hll.add(s);
                exact_counter.insert(s);
            }

            collected_data[step_idx].push_back({exact_counter.size(), hll.estimate()});
        }
    }

    std::ofstream file("hll_stats.csv");
    file << "Fraction,True_Mean,Est_Mean,Sigma,LowerBound,UpperBound\n";

    int step_idx = 0;
    for (double frac : fractions) {
        step_idx++;
        auto& runs = collected_data[step_idx];

        double sum_true = 0, sum_est = 0;
        for (const auto& d : runs) {
            sum_true += d.true_val;
            sum_est += d.est_val;
        }
        double mean_true = sum_true / runs.size();
        double mean_est = sum_est / runs.size();

        double variance = 0;
        for (const auto& d : runs) {
            variance += std::pow(d.est_val - mean_est, 2);
        }
        double sigma = std::sqrt(variance / runs.size());

        file << frac << "," 
             << mean_true << "," 
             << mean_est << "," 
             << sigma << "," 
             << (mean_est - sigma) << "," 
             << (mean_est + sigma) << "\n";
    }
    file.close();
    std::cout << "резы  в файле 'hll_stats.csv'.\n";

    return 0;
}