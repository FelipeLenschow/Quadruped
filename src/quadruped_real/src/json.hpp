#pragma once

#include <string>
#include <vector>

namespace quadruped_real::json {

// Python json.dumps formatting: floats in their shortest round-trip form (1.0, 0.439), non-ASCII escaped.
std::string str(const std::string& s);
std::string num(double v);
std::string integer(long v);
std::string boolean(bool v);
std::string list(const std::vector<std::string>& items);  // items already encoded
std::string strings(const std::vector<std::string>& items);
std::string floats(const std::vector<double>& items);

class Object {
 public:
  Object& add(const std::string& key, const std::string& encoded);
  std::string dump() const { return "{" + body_ + "}"; }

 private:
  std::string body_;
};

}  // namespace quadruped_real::json
