#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace quadruped_real::json {

std::string str(const std::string& s) {
  std::string out = "\"";
  for (size_t i = 0; i < s.size(); ++i) {
    const unsigned char c = s[i];
    char buf[16];
    if (c == '"') {
      out += "\\\"";
    } else if (c == '\\') {
      out += "\\\\";
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\t') {
      out += "\\t";
    } else if (c < 0x20) {
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else if (c < 0x80) {
      out += static_cast<char>(c);
    } else {
      // UTF-8 to code point, written as \uXXXX (surrogate pairs past the BMP).
      int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : 1;
      unsigned cp = c & (0x3F >> extra);
      for (int k = 0; k < extra && i + 1 < s.size(); ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[++i]) & 0x3F);
      if (cp >= 0x10000) {
        cp -= 0x10000;
        std::snprintf(buf, sizeof(buf), "\\u%04x\\u%04x", 0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF));
      } else {
        std::snprintf(buf, sizeof(buf), "\\u%04x", cp);
      }
      out += buf;
    }
  }
  return out + "\"";
}

std::string num(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  char buf[32];
  for (int p = 1; p <= 17; ++p) {
    std::snprintf(buf, sizeof(buf), "%.*g", p, v);
    if (std::strtod(buf, nullptr) == v) break;
  }
  std::string s = buf;
  if (s.find_first_of(".en") == std::string::npos) s += ".0";
  return s;
}

std::string integer(long v) { return std::to_string(v); }

std::string boolean(bool v) { return v ? "true" : "false"; }

std::string list(const std::vector<std::string>& items) {
  std::string out = "[";
  for (size_t i = 0; i < items.size(); ++i) out += (i ? ", " : "") + items[i];
  return out + "]";
}

std::string strings(const std::vector<std::string>& items) {
  std::vector<std::string> encoded;
  for (const auto& s : items) encoded.push_back(str(s));
  return list(encoded);
}

std::string floats(const std::vector<double>& items) {
  std::vector<std::string> encoded;
  for (double v : items) encoded.push_back(num(v));
  return list(encoded);
}

Object& Object::add(const std::string& key, const std::string& encoded) {
  body_ += (body_.empty() ? "" : ", ") + str(key) + ": " + encoded;
  return *this;
}

}  // namespace quadruped_real::json
