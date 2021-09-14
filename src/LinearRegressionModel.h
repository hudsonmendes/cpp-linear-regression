#pragma once

#include <vector>

/**
 * Models that learns through linear regression.
 */
class LinearRegressionModel
{
private:
    const int epochs;
    const double learning_rate;
    std::vector<double> losses;
    double w{0.5};
    double b{0.5};

    double train_step(const double &x,
                      const double &y_hat,
                      const double &y);

public:
    /**
     * Constructs an instance of the Logistic Regression model
     * 
     * @param learning_rate by how much we shall adjust the weights and biases
     * @param epochs the number of loops we will perform during the training time.
     */
    LinearRegressionModel(double learning_rate,
                          int epochs);

    /**
     * Trains the model using the pair X (input) and Y (standard).
     * In the process, adjusts the parameters W and b.
     * 
     * @param x the input values
     * @param y the expected/standard labels
     */
    void train(const std::vector<double> &x,
               const std::vector<double> &y);

    /**
     * Performs a prediction over input X.
     * 
     * @param x the input
     */
    double predict(const double &x);
};