#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdint.h>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>

namespace examples::utils {

struct profiling_event {
    std::string_view label;
    std::string_view category;
    std::chrono::microseconds start_time, end_time;
    size_t thread_id;
};

class profiling_session {
  private:
    inline static std::mutex s_mutex;
    inline static std::unordered_set<std::string> s_active_sessions;
    std::ofstream m_out_stream;
    const std::string m_filepath;
    int m_event_count = 0;
    std::optional<profiling_event> m_overhead_event = std::nullopt;
    std::mutex m_mutex;

    static void register_session(const std::string &filepath) {
        std::lock_guard lock(s_mutex);
        if (s_active_sessions.find(filepath) != s_active_sessions.end()) {
            throw std::runtime_error("Profiling session already exists for file: " + filepath);
        }
        s_active_sessions.insert(filepath);
    }
    static void unregister_session(const std::string &filepath) {
        std::lock_guard lock(s_mutex);
        auto erased_num = s_active_sessions.erase(filepath);
        if (erased_num == 0) {
            throw std::runtime_error("Attempted to unregister session that did not exist: " +
                                     filepath);
        }
    }

  public:
    static profiling_session &get_default() {
        static profiling_session instance("profiling.json");
        return instance;
    }

    explicit profiling_session(std::string filepath) : m_filepath(std::move(filepath)) {
        register_session(m_filepath);
        m_out_stream.open(m_filepath, std::ios::out | std::ios::trunc);
        if (!m_out_stream) {
            throw std::runtime_error("Failed to open file for profiling: " + m_filepath);
        }
        write_header();
    }
    profiling_session(const profiling_session &other) = delete;
    profiling_session &operator=(const profiling_session &other) = delete;
    profiling_session(profiling_session &&other) = delete;
    profiling_session &operator=(profiling_session &&other) = delete;
    ~profiling_session() {
        try {
            flush_overhead_event();
            write_footer();
            m_out_stream.close();
            unregister_session(m_filepath);
            std::cout << "\nRecorded profiling was saved to " << m_filepath << std::endl;
        } catch (...) {
            std::cerr << "\nError during profiling session (" << m_filepath << ") destruction" << std::endl;
        }
    }

    void add_event(const profiling_event &event, std::chrono::microseconds overhead_start) {
        std::lock_guard lock(m_mutex);
        size_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        flush_overhead_event();
        write_event(event);

        m_overhead_event = profiling_event{"overhead",
                                           "instrumentation",
                                           overhead_start,
                                           std::chrono::microseconds::zero(),
                                           thread_id};
        m_overhead_event->end_time = std::chrono::time_point_cast<std::chrono::microseconds>(
                                         std::chrono::high_resolution_clock::now())
                                         .time_since_epoch();
    }

  protected:
    void flush_overhead_event() {
        if (m_overhead_event.has_value()) {
            write_event(m_overhead_event.value());
            m_overhead_event = std::nullopt;
        }
    }

    void write_event(const profiling_event &event) {
        if (m_event_count++ > 0)
            m_out_stream << ",";

        auto start_time_us = event.start_time.count();
        auto end_time_us = event.end_time.count();
        m_out_stream << "{";
        m_out_stream << "\"name\":\"" << event.label << "\",";
        m_out_stream << "\"cat\":\"" << event.category << "\",";
        m_out_stream << "\"ph\":\"X\","; // Event phase is X - "complete event"
        m_out_stream << "\"ts\":" << start_time_us << ",";
        m_out_stream << "\"dur\":" << (end_time_us - start_time_us) << ",";
        m_out_stream << "\"pid\":0,";
        m_out_stream << "\"tid\":" << static_cast<uint32_t>(event.thread_id);
        m_out_stream << "}\n";
    }

    void write_header() { m_out_stream << "{\"traceEvents\":[\n"; }

    void write_footer() { m_out_stream << "]}"; }
};

class scope_profiler {
  public:
    scope_profiler(profiling_session &manager,
                   std::string_view label,
                   std::string_view category = "none")
        : m_manager(manager), m_label(label), m_category(category), m_finished(false) {
        m_start_time = std::chrono::high_resolution_clock::now();
    }
    scope_profiler(std::string_view label, std::string_view category = "none")
        : scope_profiler(profiling_session::get_default(), label, category) {}
    scope_profiler(const scope_profiler &) = delete;
    scope_profiler &operator=(const scope_profiler &) = delete;

    ~scope_profiler() { finish(); }

    void finish() {
        auto end_time = std::chrono::high_resolution_clock::now();
        if (m_finished) {
            return;
        }
        m_finished = true;
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_start_time)
                         .time_since_epoch();
        auto end =
            std::chrono::time_point_cast<std::chrono::microseconds>(end_time).time_since_epoch();

        size_t thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        m_manager.add_event(profiling_event{m_label, m_category, start, end, thread_id}, end);
    }

  private:
    profiling_session &m_manager;
    std::string_view m_label;
    std::string_view m_category;
    bool m_finished;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_time;
};
} // namespace examples::utils
