#pragma once

#include <string>
#include <iterator>
#include <tuple>

/**
 * Dataset reader, that outputs a tuple<x, y>.
 */
class Dataset
{
private:
    std::string file_path{};
    std::vector<std::tuple<double, double>> read() const;

public:
    /**
     * Constructs an instance of Dataset.
     * @param file_path the path to the file that must be loaded.
     */
    Dataset(const std::string &file_path);

    /**
     * Reads and generates each line of the dataset as a tuple<x, y>.
     */
    std::vector<double> x() const;

    std::vector<double> y() const;
};