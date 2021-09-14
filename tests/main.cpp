#include <iostream>

#include "../src/Dataset.h"
#include "../src/LinearRegressionModel.h"

int main()
{
    std::cout << "Logistic regression" << std::endl;

    std::cout << "1) loading data" << std::endl;
    Dataset dataset{"./data/sample.csv"};

    LinearRegressionModel model{0.01, 100};
    const auto x{12};

    std::cout << "2) training" << std::endl;
    for (auto i {0}; i < 10; i++)
    {
        std::cout << ">>> training cycle: " <<  std::to_string(i + 1) << std::endl;
        model.train(dataset.x(), dataset.y());

        const auto y_hat = model.predict(x);
        std::cout << ">>> " << std::to_string(x) << " > " << std::to_string(y_hat) << std::endl;
    }

    return 0;
}