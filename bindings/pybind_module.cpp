// Minimal pybind11 bindings exposing stub Engine
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pylittle/engine.h"
#include "pylittle/memory_manager.h"
#include "pylittle/kv_pager.h"

namespace py = pybind11;
using namespace pylittle;

PYBIND11_MODULE(_pylittle, m) {
	py::class_<GenerateResult>(m, "GenerateResult")
		.def_readwrite("text", &GenerateResult::text);

	py::class_<Engine>(m, "Engine")
		.def(py::init<>())
		.def("load_model", &Engine::load_model)
		.def("generate", &Engine::generate,
			 py::arg("prompt"), py::arg("max_tokens")=128, py::arg("temperature")=0.8f, py::arg("stream")=false)
		.def("set_safety_policy", &Engine::set_safety_policy)
		.def("get_stats", &Engine::get_stats);

	py::class_<MemoryManager>(m, "MemoryManager")
		.def(py::init<>())
		.def("has_cuda", &MemoryManager::has_cuda)
		.def("map_file", &MemoryManager::map_file, py::arg("path"))
		.def("prefetch", &MemoryManager::prefetch, py::arg("handle"), py::arg("offset"), py::arg("bytes"))
		.def("unmap", &MemoryManager::unmap, py::arg("handle"))
		.def("create_stream", &MemoryManager::create_stream)
		.def("destroy_stream", &MemoryManager::destroy_stream, py::arg("stream"))
		.def("copy_to_device_async", &MemoryManager::copy_to_device_async, py::arg("dst_device"), py::arg("src_host"), py::arg("bytes"), py::arg("stream").none(true))
		.def("synchronize_stream", &MemoryManager::synchronize_stream, py::arg("stream").none(true))
		.def("stats", &MemoryManager::stats);

	py::class_<KVPager>(m, "KVPager")
		.def(py::init<MemoryManager*, size_t>(), py::arg("mm"), py::arg("page_bytes"))
		.def("add_sequence", &KVPager::add_sequence, py::arg("seq_id"))
		.def("append_kv_bytes", &KVPager::append_kv_bytes, py::arg("seq_id"), py::arg("bytes"), py::arg("tokens"))
		.def("append_kv", [](KVPager& self, int seq_id, py::bytes data) {
			std::string s = data; self.append_kv(seq_id, s.data(), s.size());
		}, py::arg("seq_id"), py::arg("data"))
		.def("append_kv_view", [](KVPager& self, int seq_id, py::buffer b) {
			py::buffer_info info = b.request();
			size_t bytes = static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize);
			self.append_kv(seq_id, info.ptr, bytes);
		}, py::arg("seq_id"), py::arg("buffer"))
		.def("request_window", &KVPager::request_window, py::arg("seq_id"), py::arg("recent_tokens"))
		.def("evict_lru", &KVPager::evict_lru, py::arg("target_free_bytes"))
		.def("stats", &KVPager::stats);
}
