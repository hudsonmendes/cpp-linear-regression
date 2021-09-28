#include "LinearRegressionModel.h"

#include <string>
#include <iostream>
#include <fstream>

LinearRegressionModel::LinearRegressionModel(double learning_rate, int epochs, const std::string &log_path) : learning_rate(learning_rate),
                                                                                                              epochs(epochs),
                                                                                                              log_file(std::ofstream{log_path})
{
}

LinearRegressionModel::~LinearRegressionModel()
{
    if (log_file.is_open())
    {
        log_file.flush();
        log_file.close();
    }
}

void LinearRegressionModel::train(const std::vector<double> &x,
                                  const std::vector<double> &y)
{
    log_step("x", "y", "predicted", "loss", "w", "b");
    for (auto epoch{0}; epoch < epochs; ++epoch)
    {
        std::vector<double> batch_loss = {};
        for (auto i{0}; i < x.size(); ++i)
        {
            const auto y_hat = predict(x[i]);
            const auto loss = train_step(x[i], y_hat, y[i]);
            batch_loss.push_back(loss);
            losses.push_back(loss);
        }

        auto total_loss{0.0};
        for (auto i{0}; i < batch_loss.size(); i++)
            total_loss += batch_loss[i];
        std::cout << " - epoch " << std::to_string(epoch) << ", batch_loss: " << std::to_string(total_loss) << std::endl;
    }
}

double LinearRegressionModel::predict(const double &x)
{
    return (x * w) + b; // linear activation
}

double LinearRegressionModel::train_step(const double &x,
                                         const double &y_hat,
                                         const double &y)
{
    auto loss = y_hat - y;           // calculating cost
    w -= (learning_rate * loss * x); // updating w
    b -= (learning_rate * loss);     // updating b
    log_step(std::to_string(x),
             std::to_string(y),
             std::to_string(y_hat),
             std::to_string(loss),
             std::to_string(w),
             std::to_string(b));
    return loss;
}

void LinearRegressionModel::log_step(const std::string &x,
                                     const std::string &y,
                                     const std::string &y_hat,
                                     const std::string &loss,
                                     const std::string &w,
                                     const std::string &b)
{
    if (log_file.is_open())
        log_file << x << ","
                 << y << ","
                 << y_hat << ","
                 << loss << ","
                 << w << ","
                 << b << ","
                 << std::endl;
}