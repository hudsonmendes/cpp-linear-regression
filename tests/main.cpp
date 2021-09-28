#include <iostream>
#include <iomanip>

#include "../src/Dataset.h"
#include "../src/LinearRegressionModel.h"

int main()
{
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Logistic regression" << std::endl;

    std::cout << "1) loading data" << std::endl;
    Dataset dataset{"./data/sample.csv"};
    std::cout << " - loaded " << std::to_string(dataset.x().size()) << " data points" << std::endl;

    std::cout << "2) training" << std::endl;
    LinearRegressionModel model{0.0000001, 10, "data/log.csv"};
    model.train(dataset.x(), dataset.y());

    std::cout << "3) predict" << std::endl;
    std::cout << " > enter the year: ";
    unsigned short year{0};
    std::cin >> year;
    auto predicted_price{model.predict(year)};
    std::cout << "  - price predicted " << std::to_string(predicted_price) << std::endl;

    return 0;
}