#pragma once
#include <memory>
#include <string>
#include <vector>

namespace pylittle {

struct GenerateResult {
    std::string text;
};

class Engine {
public:
    Engine();
    ~Engine();

    bool load_model(const std::string& path, const std::string& device = "auto");

    GenerateResult generate(const std::string& prompt,
                            int max_tokens = 128,
                            float temperature = 0.8f,
                            bool stream = false);

    void set_safety_policy(int max_gpu_temp, int max_gpu_util);

    std::string get_stats() const;
};

} // namespace pylittle
