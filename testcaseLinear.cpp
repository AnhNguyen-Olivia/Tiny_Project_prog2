#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "matrix.h"
#include "vector.h"
#include "nonSquareLinearSystem.h"

// Struct to hold a data instance with statistics for normalization
struct DataEntry {
    double MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX, PRP;
};

struct FeatureStats {
    double mean;
    double stddev;
};

// Function to load data from CSV with error handling
std::vector<DataEntry> loadData(const std::string& filename) {
    std::vector<DataEntry> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + filename);
    }

    std::string line;
    int line_num = 0;

    while (std::getline(file, line)) {
        line_num++;
        std::stringstream ss(line);
        std::string token;
        DataEntry entry;

        try {
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
        } catch (const std::exception& e) {
            std::cerr << "Warning: Error parsing line " << line_num << ": " << e.what() << std::endl;
        }
    }

    if (data.empty()) {
        throw std::runtime_error("Error: No valid data loaded from file");
    }

    return data;
}

// Calculate feature statistics for normalization
void calculateFeatureStats(const std::vector<DataEntry>& data, 
                         std::vector<FeatureStats>& stats) {
    if (data.empty()) return;

    // Initialize sums
    std::vector<double> sums(6, 0.0);
    std::vector<double> sq_sums(6, 0.0);

    // Calculate sums
    for (const auto& entry : data) {
        sums[0] += entry.MYCT;
        sums[1] += entry.MMIN;
        sums[2] += entry.MMAX;
        sums[3] += entry.CACH;
        sums[4] += entry.CHMIN;
        sums[5] += entry.CHMAX;

        sq_sums[0] += entry.MYCT * entry.MYCT;
        sq_sums[1] += entry.MMIN * entry.MMIN;
        sq_sums[2] += entry.MMAX * entry.MMAX;
        sq_sums[3] += entry.CACH * entry.CACH;
        sq_sums[4] += entry.CHMIN * entry.CHMIN;
        sq_sums[5] += entry.CHMAX * entry.CHMAX;
    }

    // Calculate mean and stddev
    double n = data.size();
    stats.resize(6);
    for (int i = 0; i < 6; ++i) {
        stats[i].mean = sums[i] / n;
        double variance = (sq_sums[i] / n) - (stats[i].mean * stats[i].mean);
        stats[i].stddev = std::sqrt(variance);
        
        // Handle case where stddev is 0 (constant feature)
        if (stats[i].stddev < 1e-10) stats[i].stddev = 1.0;
    }
}

// Normalize features using calculated statistics
void normalizeFeatures(std::vector<DataEntry>& data, 
                     const std::vector<FeatureStats>& stats) {
    for (auto& entry : data) {
        entry.MYCT = (entry.MYCT - stats[0].mean) / stats[0].stddev;
        entry.MMIN = (entry.MMIN - stats[1].mean) / stats[1].stddev;
        entry.MMAX = (entry.MMAX - stats[2].mean) / stats[2].stddev;
        entry.CACH = (entry.CACH - stats[3].mean) / stats[3].stddev;
        entry.CHMIN = (entry.CHMIN - stats[4].mean) / stats[4].stddev;
        entry.CHMAX = (entry.CHMAX - stats[5].mean) / stats[5].stddev;
    }
}

// Split data into training and test sets with randomization
void trainTestSplit(const std::vector<DataEntry>& data, 
                   std::vector<DataEntry>& train,
                   std::vector<DataEntry>& test,
                   double testSize = 0.2, 
                   unsigned int randomSeed = 42) {
    if (data.empty()) return;

    // Create a shuffled index
    std::vector<size_t> indices(data.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    
    // Simple shuffle using seed for reproducibility
    std::srand(randomSeed);
    std::random_shuffle(indices.begin(), indices.end());

    // Calculate split point
    size_t split = static_cast<size_t>(data.size() * (1.0 - testSize));

    // Split the data
    train.clear();
    test.clear();
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i < split) {
            train.push_back(data[indices[i]]);
        } else {
            test.push_back(data[indices[i]]);
        }
    }
}

// Calculate RMSE
double calculateRMSE(const Vector& predictions, const Vector& actual) {
    if (predictions.getSize() != actual.getSize() || predictions.getSize() == 0) {
        throw std::invalid_argument("Vectors must be of same non-zero size");
    }

    double sum = 0.0;
    for (int i = 0; i < predictions.getSize(); ++i) {
        double diff = predictions[i] - actual[i];
        sum += diff * diff;
    }
    return std::sqrt(sum / predictions.getSize());
}

// Calculate R-squared coefficient
double calculateRSquared(const Vector& predictions, const Vector& actual) {
    if (predictions.getSize() != actual.getSize() || predictions.getSize() == 0) {
        throw std::invalid_argument("Vectors must be of same non-zero size");
    }

    double actual_mean = 0.0;
    for (int i = 0; i < actual.getSize(); ++i) {
        actual_mean += actual[i];
    }
    actual_mean /= actual.getSize();

    double ss_total = 0.0;
    double ss_res = 0.0;
    for (int i = 0; i < predictions.getSize(); ++i) {
        ss_total += (actual[i] - actual_mean) * (actual[i] - actual_mean);
        ss_res += (actual[i] - predictions[i]) * (actual[i] - predictions[i]);
    }

    return 1.0 - (ss_res / ss_total);
}

int main() {
    try {
        // Load and prepare data
        auto data = loadData("machine.data");
        
        // Calculate and apply feature normalization
        std::vector<FeatureStats> stats;
        calculateFeatureStats(data, stats);
        normalizeFeatures(data, stats);
        
        // Split into train/test with randomization
        std::vector<DataEntry> trainData, testData;
        trainTestSplit(data, trainData, testData, 0.2, 42);

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

        // Solve using different methods
        NonSquareLinearSystem solver(A, b);
        
        // Method 1: Pseudo-inverse
        Vector coefficients_pinv = solver.SolveWithPseudoInverse();
        
        // Method 2: Tikhonov regularization
        double lambda = 0.1; // regularization parameter
        Vector coefficients_tikhonov = solver.SolveWithTikhonov(lambda);

        // Evaluate on test set
        Vector testPredictions_pinv(testData.size());
        Vector testPredictions_tikhonov(testData.size());
        Vector testActual(testData.size());
        
        for (size_t i = 0; i < testData.size(); ++i) {
            const auto& entry = testData[i];
            testActual[i] = entry.PRP;
            
            // Pseudo-inverse prediction
            testPredictions_pinv[i] = coefficients_pinv[0] * entry.MYCT +
                                     coefficients_pinv[1] * entry.MMIN +
                                     coefficients_pinv[2] * entry.MMAX +
                                     coefficients_pinv[3] * entry.CACH +
                                     coefficients_pinv[4] * entry.CHMIN +
                                     coefficients_pinv[5] * entry.CHMAX;
            
            // Tikhonov prediction
            testPredictions_tikhonov[i] = coefficients_tikhonov[0] * entry.MYCT +
                                        coefficients_tikhonov[1] * entry.MMIN +
                                        coefficients_tikhonov[2] * entry.MMAX +
                                        coefficients_tikhonov[3] * entry.CACH +
                                        coefficients_tikhonov[4] * entry.CHMIN +
                                        coefficients_tikhonov[5] * entry.CHMAX;
        }

        // Calculate and print metrics
        double rmse_pinv = calculateRMSE(testPredictions_pinv, testActual);
        double r2_pinv = calculateRSquared(testPredictions_pinv, testActual);
        
        double rmse_tikhonov = calculateRMSE(testPredictions_tikhonov, testActual);
        double r2_tikhonov = calculateRSquared(testPredictions_tikhonov, testActual);

        std::cout << "=== Pseudo-Inverse Solution ===\n";
        std::cout << "Coefficients:\n";
        coefficients_pinv.Print();
        std::cout << "Test RMSE: " << rmse_pinv << std::endl;
        std::cout << "Test R-squared: " << r2_pinv << std::endl;

        std::cout << "\n=== Tikhonov Regularization (lambda=" << lambda << ") ===\n";
        std::cout << "Coefficients:\n";
        coefficients_tikhonov.Print();
        std::cout << "Test RMSE: " << rmse_tikhonov << std::endl;
        std::cout << "Test R-squared: " << r2_tikhonov << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
