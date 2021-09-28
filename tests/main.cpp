#include <iostream>
#include <iomanip>

#include "../src/Dataset.h"
#include "../src/LinearRegressionModel.h"

int main()
{
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Logistic regression" << std::endl;

    std::cout << "1) hyper parameters" << std::endl;
    const auto learning_rate{0.00000005};
    const auto epochs{20};
    std::cout << " - learning_rate: " << std::to_string(learning_rate) << std::endl;
    std::cout << " -        epochs: " << std::to_string(epochs) << std::endl;

    std::cout << "2) loading data" << std::endl;
    Dataset dataset{"./data/sample.csv"};
    std::cout << " - loaded " << std::to_string(dataset.x().size()) << " data points" << std::endl;

    std::cout << "3) training" << std::endl;
    LinearRegressionModel model{learning_rate, epochs, "data/log.csv"};
    model.train(dataset.x(), dataset.y());

    std::cout << "4) predict" << std::endl;
    std::cout << " > enter the year: ";
    unsigned short year{0};
    std::cin >> year;
    auto predicted_price{model.predict(year)};
    std::cout << "  - price predicted " << std::to_string(predicted_price) << std::endl;

    return 0;
}