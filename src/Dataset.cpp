#include "Dataset.h"

#include <vector>
#include <iostream>
#include <fstream>

#define DELIMITER ","

Dataset::Dataset(const std::string &file_path) : file_path(file_path)
{
}

std::vector<double> Dataset::x() const
{
    std::vector<double> x{};
    for (const auto &[xi, _] : read())
        x.push_back(xi);
    return x;
}

std::vector<double> Dataset::y() const
{
    std::vector<double> y{};
    for (const auto &[_, yi] : read())
        y.push_back(yi);
    return y;
}

std::vector<std::tuple<double, double>> Dataset::read() const
{
    std::vector<std::tuple<double, double>> out{};
    std::ifstream file(file_path);
    std::string str;
    while (std::getline(file, str))
    {
        const auto x = str.substr(0, str.find(DELIMITER));
        const auto y = str.substr(str.find(DELIMITER) + 1);
        if (x == "x" || y == "y")
            continue;
        const auto pair = std::tuple<double, double>{std::stod(x), std::stod(y)};
        out.push_back(pair);
    }
    return out;
}