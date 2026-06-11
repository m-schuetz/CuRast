// compat_print.h -- std::print/println polyfill for C++23
// Include this instead of <print> for portability across compilers
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>

#pragma once

#include <format>
#include <iostream>
#include <stacktrace>

// GCC 13 doesn't have std::formatter for std::stacktrace, provide overloads
#if !defined(__cpp_lib_stacktrace_from_string) || __cpp_lib_stacktrace_from_string < 202011L

namespace std {
    // Direct stacktrace printing (not via format)
    inline void println(const basic_stacktrace<allocator<stacktrace_entry>>& st) {
        cout << to_string(st) << '\n';
    }
    // Format string "{}" with stacktrace: call to_string ourselves
    inline void println(string_view fmt, const basic_stacktrace<allocator<stacktrace_entry>>& st) {
        // If format string is just "{}", print the stacktrace directly
        if (fmt == "{}") {
            cout << to_string(st) << '\n';
        } else {
            // For more complex format strings, just print as-is
            cout << to_string(st) << '\n';
        }
    }
}
#endif

#if __has_include(<print>)
#include <print>
#else
// Minimal std::print/println implementation using std::format
namespace std {
    template<typename... Args>
    void print(format_string<Args...> fmt, Args&&... args) {
        cout << format(fmt, forward<Args>(args)...);
    }
    template<typename... Args>
    void println(format_string<Args...> fmt, Args&&... args) {
        cout << format(fmt, forward<Args>(args)...) << '\n';
    }
    inline void println() { cout << '\n'; }

    // Output stream versions
    template<typename... Args>
    void print(ostream& os, format_string<Args...> fmt, Args&&... args) {
        os << format(fmt, forward<Args>(args)...);
    }
    template<typename... Args>
    void println(ostream& os, format_string<Args...> fmt, Args&&... args) {
        os << format(fmt, forward<Args>(args)...) << '\n';
    }
}
#endif
