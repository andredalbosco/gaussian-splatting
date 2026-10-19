#include "utils.hpp"

#include <fstream>
#include <iterator>

std::string readFile(const char* path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
