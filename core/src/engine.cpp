#include "pylittle/engine.h"
#include <sstream>

namespace pylittle {

Engine::Engine() {}
Engine::~Engine() {}

bool Engine::load_model(const std::string& path, const std::string& device) {
    (void)path; (void)device; // placeholder
    return true;
}

GenerateResult Engine::generate(const std::string& prompt, int max_tokens, float temperature, bool stream) {
    (void)max_tokens; (void)temperature; (void)stream; // placeholder
    GenerateResult r;
    r.text = std::string("[STUB] ") + prompt;
    return r;
}

void Engine::set_safety_policy(int max_gpu_temp, int max_gpu_util) {
    (void)max_gpu_temp; (void)max_gpu_util; // placeholder
}

std::string Engine::get_stats() const {
    std::ostringstream oss;
    oss << "pylittle stub engine";
    return oss.str();
}

} // namespace pylittle
