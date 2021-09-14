#include "LinearRegressionModel.h"

#include <string>
#include <iostream>

LinearRegressionModel::LinearRegressionModel(double learning_rate, int epochs) : learning_rate(learning_rate),
                                                                                 epochs(epochs)
{
}

void LinearRegressionModel::train(const std::vector<double> &x,
                                  const std::vector<double> &y)
{
    for (auto epoch = 0; epoch < epochs; ++epoch)
    {
        std::vector<double> batch_loss = {};
        for (auto i = 0; i < x.size(); ++i)
        {
            const auto y_hat = predict(x[i]);
            const auto loss = train_step(x[i], y_hat, y[i]);
            batch_loss.push_back(loss);
            losses.push_back(loss);
        }
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
    return loss;
}