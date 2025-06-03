#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include "matrix.h"
#include "vector.h"
#include "nonSquareLinearSystem.h"

// Struct to hold a data instance
struct DataEntry {
    double MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX, PRP;
};

// Function to load data from CSV
std::vector<DataEntry> loadData(const std::string& filename) {
    std::vector<DataEntry> data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        DataEntry entry;

        // Skip first two columns (non-predictive)
        for (int i = 0; i < 2; ++i) std::getline(ss, token, ',');

        // Read predictive features
        std::getline(ss, token, ','); entry.MYCT = std::stod(token);
        std::getline(ss, token, ','); entry.MMIN = std::stod(token);
        std::getline(ss, token, ','); entry.MMAX = std::stod(token);
        std::getline(ss, token, ','); entry.CACH = std::stod(token);
        std::getline(ss, token, ','); entry.CHMIN = std::stod(token);
        std::getline(ss, token, ','); entry.CHMAX = std::stod(token);

        // Read target (PRP)
        std::getline(ss, token, ','); entry.PRP = std::stod(token);

        data.push_back(entry);
    }

    return data;
}

// Split data into training and test sets
void trainTestSplit(const std::vector<DataEntry>& data, 
                   std::vector<DataEntry>& train,
                   std::vector<DataEntry>& test,
                   double testSize = 0.2) {
    int n = data.size();
    int testSamples = static_cast<int>(n * testSize);
    
    // Use first 80% for training, last 20% for testing
    train = std::vector<DataEntry>(data.begin(), data.end() - testSamples);
    test = std::vector<DataEntry>(data.end() - testSamples, data.end());
}

// Calculate RMSE
double calculateRMSE(const Vector& predictions, const Vector& actual) {
    double sum = 0.0;
    for (int i = 0; i < predictions.getSize(); ++i) {
        double diff = predictions[i] - actual[i];
        sum += diff * diff;
    }
    return std::sqrt(sum / predictions.getSize());
}

int main() {
    // Load data
    auto data = loadData("machine.data");
    
    // Split into train/test
    std::vector<DataEntry> trainData, testData;
    trainTestSplit(data, trainData, testData);

    // Build matrix A and vector b for training
    Matrix A(trainData.size(), 6);
    Vector b(trainData.size());
    
    for (size_t i = 0; i < trainData.size(); ++i) {
        const auto& entry = trainData[i];
        A(i+1, 1) = entry.MYCT;
        A(i+1, 2) = entry.MMIN;
        A(i+1, 3) = entry.MMAX;
        A(i+1, 4) = entry.CACH;
        A(i+1, 5) = entry.CHMIN;
        A(i+1, 6) = entry.CHMAX;
        b[i] = entry.PRP;
    }

    // Solve using pseudo-inverse
    NonSquareLinearSystem solver(A, b);
    Vector coefficients = solver.SolveWithPseudoInverse();

    // Evaluate on test set
    Vector testPredictions(testData.size());
    Vector testActual(testData.size());
    
    for (size_t i = 0; i < testData.size(); ++i) {
        const auto& entry = testData[i];
        testActual[i] = entry.PRP;
        
        double prediction = coefficients[0] * entry.MYCT +
                          coefficients[1] * entry.MMIN +
                          coefficients[2] * entry.MMAX +
                          coefficients[3] * entry.CACH +
                          coefficients[4] * entry.CHMIN +
                          coefficients[5] * entry.CHMAX;
        testPredictions[i] = prediction;
    }

    // Calculate and print RMSE
    double rmse = calculateRMSE(testPredictions, testActual);
    std::cout << "Coefficients:\n";
    coefficients.Print();
    std::cout << "\nTest RMSE: " << rmse << std::endl;

    return 0;
}