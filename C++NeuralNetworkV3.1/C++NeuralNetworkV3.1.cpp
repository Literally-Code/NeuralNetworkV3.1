#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>

#include "Network.h"


int main()
{
    // Local variables
    const std::string dataLocation = ".\\image\\";
    const std::string trainingFile = dataLocation + "data.csv";
    std::vector<std::string> tokens;
    std::vector<std::string> fileNames;
    std::vector<int> values;
    int pairSize = 0;
    std::ifstream dataFile(trainingFile);
    std::string lineBuffer;
    std::string token;
    std::vector<double> expectedOutput = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    Network net = Network(IMAGE_SIZE * IMAGE_SIZE, 2, 16, 10);
    if (!dataFile.is_open()) {
        std::cerr << "Error: Failed to open the file " << trainingFile << std::endl;
        return 1; // Exit with an error code
    }

    while (std::getline(dataFile, lineBuffer))
    {
        std::istringstream iss(lineBuffer);

        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }

        fileNames.push_back(dataLocation + tokens[0]);
        values.push_back(std::stoi(tokens[1]));
        pairSize++;
        tokens.clear();
    }

    for (int epoch = 0; epoch < 100; epoch++)
    {
        std::cout << "Epoch: " << epoch << std::endl;
        for (int file = 0; file < pairSize; file++)
        {
            for (int i = 0; i < 10; i++)
            {
                expectedOutput[i] = 0;
            }

            net.inputPng(fileNames[file].c_str());
            expectedOutput[values[file]] = 1;

            net.activate();
            net.backPropagate(expectedOutput);

        }
        net.inputPng(fileNames[0].c_str());

        net.activate();

        std::cout << fileNames[0] << " " << std::endl;
        net.Cost();
    }

    // Test

    for (int file = 0; file < 1; file++)
    {
        net.inputPng(fileNames[file].c_str());

        net.activate();

        std::cout << fileNames[file] << " " << std::endl;
        net.Cost();
    }

    // net.visualize();
    char c;
    std::cin >> c;
}