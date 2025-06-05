#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>  // For output formatting
#include <chrono>   // For timing operations
#include "matrix.h"
#include "vector.h"
#include "nonSquareLinearSystem.h"

// ANSI color codes for output
const std::string GREEN = "\033[32m";
const std::string RED = "\033[31m";
const std::string YELLOW = "\033[33m";
const std::string BLUE = "\033[34m";
const std::string MAGENTA = "\033[35m";
const std::string CYAN = "\033[36m";
const std::string BOLD = "\033[1m";
const std::string RESET = "\033[0m";

// Struct to hold a data instance
struct DataEntry {
    double MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX, PRP;
};

// Progress bar function with enhanced visual
void showProgressBar(int progress, int total) {
    const int barWidth = 50;
    float percentage = static_cast<float>(progress) / total;
    int pos = static_cast<int>(barWidth * percentage);
    
    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << CYAN << "=" << RESET;
        else if (i == pos) std::cout << BLUE << ">" << RESET;
        else std::cout << " ";
    }
    std::cout << "] " << BOLD << int(percentage * 100.0) << "%" << RESET << "\r";
    std::cout.flush();
    
    if (progress == total) {
        std::cout << std::endl;
    }
}

// Print section header with consistent styling
void printHeader(const std::string& title) {
    const int WIDTH = 70;
    std::cout << "\n" << BLUE << std::string(WIDTH, '=') << RESET << std::endl;
    std::cout << BOLD << "  " << title << RESET << std::endl;
    std::cout << BLUE << std::string(WIDTH, '=') << RESET << std::endl;
}

// Function to load data from CSV with progress indication
std::vector<DataEntry> loadData(const std::string& filename) {
    std::vector<DataEntry> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << RED << "ERROR: Could not open file " << filename << RESET << std::endl;
        return data;
    }
    
    printHeader("DATA LOADING");
    std::cout << "Source: " << CYAN << filename << RESET << std::endl;
    
    // Count lines for progress bar
    int totalLines = 0;
    std::string line;
    std::ifstream countFile(filename);
    while (std::getline(countFile, line)) totalLines++;
    countFile.close();
    
    int currentLine = 0;
    while (std::getline(file, line)) {
        // Show progress
        if (totalLines > 100 && currentLine % (totalLines/100) == 0) {
            showProgressBar(currentLine, totalLines);
        }
        
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
        currentLine++;
    }

    if (totalLines > 100) showProgressBar(totalLines, totalLines);
    std::cout << GREEN << "[SUCCESS] " << RESET << "Loaded " << BOLD << data.size() 
              << RESET << " data entries" << std::endl;
    return data;
}

// Structure to hold normalization parameters
struct NormParams {
    double mean;
    double std;
    double min;
    double max;
};

// Function that implements Gaussian Elimination with partial pivoting
Vector gaussianElimination(const Matrix& A, const Vector& b) {
    int n = A.GetNumRows();
    
    // Create copies we can modify
    Matrix Acopy = A;
    Vector bcopy = b;
    Vector x(n);
    
    // Forward elimination with partial pivoting
    for (int k = 1; k <= n - 1; k++) {
        // Find pivot
        int maxRow = k;
        double maxVal = std::abs(Acopy(k, k));
        for (int i = k + 1; i <= n; i++) {
            if (std::abs(Acopy(i, k)) > maxVal) {
                maxVal = std::abs(Acopy(i, k));
                maxRow = i;
            }
        }
        
        // Check for singularity
        if (maxVal < 1e-10) {
            std::cerr << RED << "Warning: Matrix may be singular or ill-conditioned" << RESET << std::endl;
            return x;
        }
        
        // Swap rows if needed
        if (maxRow != k) {
            for (int j = k; j <= n; j++) {
                double temp = Acopy(k, j);
                Acopy(k, j) = Acopy(maxRow, j);
                Acopy(maxRow, j) = temp;
            }
            double temp = bcopy(k);
            bcopy(k) = bcopy(maxRow);
            bcopy(maxRow) = temp;
        }
        
        // Eliminate below
        for (int i = k + 1; i <= n; i++) {
            double factor = Acopy(i, k) / Acopy(k, k);
            for (int j = k; j <= n; j++) {
                Acopy(i, j) -= factor * Acopy(k, j);
            }
            bcopy(i) -= factor * bcopy(k);
        }
    }
    
    // Back substitution
    for (int i = n; i >= 1; i--) {
        double sum = 0.0;
        for (int j = i + 1; j <= n; j++) {
            sum += Acopy(i, j) * x(j);
        }
        x(i) = (bcopy(i) - sum) / Acopy(i, i);
    }
    
    return x;
}

// Function to normalize a dataset with enhanced feedback
std::vector<NormParams> normalizeData(std::vector<DataEntry>& data, bool useMaxNorm = false) {
    printHeader("DATA NORMALIZATION");
    
    std::vector<NormParams> params(6); // 6 features
    std::vector<double> sums(6, 0.0);
    std::vector<double> sumSquares(6, 0.0);
    std::vector<double> mins(6, std::numeric_limits<double>::max());
    std::vector<double> maxs(6, std::numeric_limits<double>::lowest());
    
    // Calculate sums, mins and maxs
    for (const auto& entry : data) {
        std::vector<double> features = {
            entry.MYCT, entry.MMIN, entry.MMAX, 
            entry.CACH, entry.CHMIN, entry.CHMAX
        };
        
        for (int i = 0; i < 6; i++) {
            sums[i] += features[i];
            sumSquares[i] += features[i] * features[i];
            mins[i] = std::min(mins[i], features[i]);
            maxs[i] = std::max(maxs[i], features[i]);
        }
    }
    
    // Calculate means and standard deviations
    int n = data.size();
    for (int i = 0; i < 6; i++) {
        params[i].mean = sums[i] / n;
        params[i].std = sqrt((sumSquares[i] / n) - (params[i].mean * params[i].mean));
        params[i].min = mins[i];
        params[i].max = maxs[i];
        if (params[i].std == 0) params[i].std = 1; // Prevent division by zero
    }
    
    // Apply normalization
    std::cout << "Using " << (useMaxNorm ? "max-value" : "z-score") << " normalization..." << std::endl;
    
    for (auto& entry : data) {
        if (useMaxNorm) {
            // Max-value normalization (feature / max_value)
            entry.MYCT = entry.MYCT / params[0].max;
            entry.MMIN = entry.MMIN / params[1].max;
            entry.MMAX = entry.MMAX / params[2].max;
            entry.CACH = entry.CACH / params[3].max;
            entry.CHMIN = entry.CHMIN / params[4].max;
            entry.CHMAX = entry.CHMAX / params[5].max;
        } else {
            // Z-score normalization ((feature - mean) / std)
            entry.MYCT = (entry.MYCT - params[0].mean) / params[0].std;
            entry.MMIN = (entry.MMIN - params[1].mean) / params[1].std;
            entry.MMAX = (entry.MMAX - params[2].mean) / params[2].std;
            entry.CACH = (entry.CACH - params[3].mean) / params[3].std;
            entry.CHMIN = (entry.CHMIN - params[4].mean) / params[4].std;
            entry.CHMAX = (entry.CHMAX - params[5].mean) / params[5].std;
        }
    }
    
    // Print normalization info
    std::cout << BOLD << "Feature Statistics:" << RESET << std::endl;
    std::cout << CYAN << std::setw(10) << "Feature" << std::setw(12) << "Mean" << std::setw(12) << "StdDev" 
              << std::setw(12) << "Min" << std::setw(12) << "Max" << RESET << std::endl;
    std::cout << std::string(58, '-') << std::endl;
    
    std::vector<std::string> featureNames = {"MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"};
    for (int i = 0; i < 6; i++) {
        std::cout << YELLOW << std::setw(10) << featureNames[i] << RESET
                  << std::setw(12) << std::fixed << std::setprecision(4) << params[i].mean 
                  << std::setw(12) << params[i].std
                  << std::setw(12) << params[i].min
                  << std::setw(12) << params[i].max << std::endl;
    }
    std::cout << GREEN << "[SUCCESS] " << RESET << "Normalization complete" << std::endl;
    
    return params;
}

// Split data into training and test sets with random shuffling
void trainTestSplit(std::vector<DataEntry>& data, 
                   std::vector<DataEntry>& train,
                   std::vector<DataEntry>& test,
                   double testSize = 0.2) {
    printHeader("DATASET SPLITTING");
    
    // Shuffle the data
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
    
    int n = data.size();
    int testSamples = static_cast<int>(n * testSize);
    
    train = std::vector<DataEntry>(data.begin(), data.end() - testSamples);
    test = std::vector<DataEntry>(data.end() - testSamples, data.end());
    
    std::cout << GREEN << "[SUCCESS] " << RESET << "Data split into:" << std::endl;
    std::cout << "  - " << CYAN << "Training: " << RESET << BOLD << train.size() << RESET << " samples" << std::endl;
    std::cout << "  - " << YELLOW << "Testing:  " << RESET << BOLD << test.size() << RESET << " samples" << std::endl;
    std::cout << "  - " << MAGENTA << "Ratio:    " << RESET << std::fixed << std::setprecision(1) 
              << (100 - testSize * 100) << "% / " << (testSize * 100) << "%" << std::endl;
}

// Calculate RMSE and other metrics
struct ModelMetrics {
    double rmse;
    double mae;
    double r2;
};

ModelMetrics calculateMetrics(const Vector& predictions, const Vector& actual) {
    ModelMetrics metrics;
    double sumSquaredError = 0.0;
    double sumAbsError = 0.0;
    double sumActual = 0.0;
    int n = predictions.getSize();
    int validPredictions = 0;
    
    for (int i = 1; i <= n; ++i) {
        // Skip NaN values
        if (std::isnan(predictions(i)) || std::isnan(actual(i))) {
            continue;
        }
        
        double diff = predictions(i) - actual(i);
        sumSquaredError += diff * diff;
        sumAbsError += std::abs(diff);
        sumActual += actual(i);
        validPredictions++;
    }
    
    // Prevent division by zero if all predictions are NaN
    if (validPredictions == 0) {
        metrics.rmse = std::numeric_limits<double>::quiet_NaN();
        metrics.mae = std::numeric_limits<double>::quiet_NaN();
        metrics.r2 = std::numeric_limits<double>::quiet_NaN();
        return metrics;
    }
    
    // Calculate metrics
    metrics.rmse = std::sqrt(sumSquaredError / validPredictions);
    metrics.mae = sumAbsError / validPredictions;
    
    // Calculate R-squared
    double meanActual = sumActual / validPredictions;
    double totalSumSquares = 0.0;
    for (int i = 1; i <= n; ++i) {
        if (std::isnan(actual(i))) continue;
        totalSumSquares += std::pow(actual(i) - meanActual, 2);
    }
    
    if (totalSumSquares < 1e-10) {
        metrics.r2 = 0.0;  // Prevent division by very small number
    } else {
        metrics.r2 = 1.0 - (sumSquaredError / totalSumSquares);
    }
    
    return metrics;
}

void printModelSummary(const Vector& coefficients, 
                      const std::vector<NormParams>& normParams,
                      const ModelMetrics& metrics,
                      bool useMaxNorm = false,
                      const std::string& methodName = "Pseudo-Inverse") {
    printHeader("MODEL SUMMARY: " + methodName);
    
    // Print metrics
    std::cout << BOLD << "Model Performance Metrics:" << RESET << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << std::setw(20) << "Metric" << std::setw(20) << "Value" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    // Handle NaN values in metrics
    if (std::isnan(metrics.rmse)) {
        std::cout << CYAN << std::setw(20) << "RMSE:" << RESET << std::setw(20) << "Error" << std::endl;
    } else {
        std::cout << CYAN << std::setw(20) << "RMSE:" << RESET << std::setw(20) << std::fixed << std::setprecision(4) << metrics.rmse << std::endl;
    }
    
    if (std::isnan(metrics.mae)) {
        std::cout << CYAN << std::setw(20) << "MAE:" << RESET << std::setw(20) << "Error" << std::endl;
    } else {
        std::cout << CYAN << std::setw(20) << "MAE:" << RESET << std::setw(20) << std::fixed << std::setprecision(4) << metrics.mae << std::endl;
    }
    
    if (std::isnan(metrics.r2)) {
        std::cout << CYAN << std::setw(20) << "R^2:" << RESET << std::setw(20) << "Error" << std::endl;
    } else {
        std::cout << CYAN << std::setw(20) << "R^2:" << RESET << std::setw(20) << std::fixed << std::setprecision(4) << metrics.r2 << std::endl;
    }
    
    std::cout << std::string(40, '-') << std::endl << std::endl;
    
    // Check if coefficients contain NaN values
    bool hasNaN = false;
    for (int i = 1; i <= coefficients.getSize(); i++) {
        if (std::isnan(coefficients(i))) {
            hasNaN = true;
            break;
        }
    }
    
    if (hasNaN) {
        std::cout << RED << "[ERROR] " << RESET << "Model coefficients contain NaN values. Unable to display model details." << std::endl;
        return;
    }
    
    // Print coefficients
    std::cout << BOLD << "Model Coefficients (Normalized Scale):" << RESET << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << std::setw(20) << "Parameter" << std::setw(20) << "Value" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << MAGENTA << std::setw(20) << "Intercept:" << RESET << std::setw(20) << std::fixed << std::setprecision(6) << coefficients(1) << std::endl;
    
    std::vector<std::string> featureNames = {"MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"};
    for (int i = 0; i < 6; i++) {
        std::cout << MAGENTA << std::setw(20) << featureNames[i] + ":" << RESET << std::setw(20) << std::fixed << std::setprecision(6) << coefficients(i+2) << std::endl;
    }
    std::cout << std::string(40, '-') << std::endl << std::endl;
    
    // Print original scale coefficients
    std::cout << BOLD << "Model Coefficients (Original Scale):" << RESET << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << std::setw(20) << "Parameter" << std::setw(20) << "Value" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    double originalIntercept = coefficients(1);
    for (int i = 0; i < 6; i++) {
        if (useMaxNorm) {
            originalIntercept -= coefficients(i+2) * 0; // For max normalization, no mean adjustment
        } else {
            originalIntercept -= coefficients(i+2) * normParams[i].mean / normParams[i].std;
        }
    }
    std::cout << YELLOW << std::setw(20) << "Intercept:" << RESET << std::setw(20) << std::fixed << std::setprecision(6) << originalIntercept << std::endl;
    
    for (int i = 0; i < 6; i++) {
        double originalCoef;
        if (useMaxNorm) {
            // FIXED: Changed 'params' to 'normParams'
            originalCoef = coefficients(i+2) / normParams[i].max;
        } else {
            originalCoef = coefficients(i+2) / normParams[i].std;
        }
        std::cout << YELLOW << std::setw(20) << featureNames[i] + ":" << RESET << std::setw(20) << std::fixed << std::setprecision(6) << originalCoef << std::endl;
    }
    std::cout << std::string(40, '-') << std::endl << std::endl;
    
    // Print the linear regression equation
    std::cout << BOLD << "Linear Regression Equation:" << RESET << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << BLUE << "PRP = " << std::fixed << std::setprecision(4) << originalIntercept;
    for (int i = 0; i < 6; i++) {
        double coef;
        if (useMaxNorm) {
            // FIXED: Changed 'params' to 'normParams'
            coef = coefficients(i+2) / normParams[i].max;
        } else {
            coef = coefficients(i+2) / normParams[i].std;
        }
        if (coef >= 0) std::cout << " + ";
        else std::cout << " - ";
        std::cout << std::abs(coef) << " * " << featureNames[i];
    }
    std::cout << RESET << std::endl;
    std::cout << std::string(70, '-') << std::endl;
}

// Display application version and runtime info
void displayVersionInfo() {
    const std::string VERSION = "v1.1.0";
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    
    std::cout << BLUE << "\n" << std::string(70, '*') << RESET << std::endl;
    std::cout << BOLD << "COMPUTER HARDWARE PERFORMANCE PREDICTION " << VERSION << RESET << std::endl;
    std::cout << "Execution date: " << std::ctime(&now_c);
    std::cout << BLUE << std::string(70, '*') << RESET << std::endl;
}

// Compare different solution methods
void compareMethodsTable(const std::vector<ModelMetrics>& allMetrics, const std::vector<std::string>& methodNames) {
    printHeader("SOLUTION METHODS COMPARISON");
    
    std::cout << BOLD << std::setw(25) << "Method" << std::setw(15) << "RMSE" 
              << std::setw(15) << "MAE" << std::setw(15) << "R^2" << RESET << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (size_t i = 0; i < methodNames.size(); i++) {
        std::string color = (i == 0) ? CYAN : (i == 1) ? YELLOW : MAGENTA;
        std::cout << color << std::setw(25) << methodNames[i] << RESET
                  << std::setw(15) << std::fixed << std::setprecision(4) << allMetrics[i].rmse
                  << std::setw(15) << std::fixed << std::setprecision(4) << allMetrics[i].mae
                  << std::setw(15) << std::fixed << std::setprecision(4) << allMetrics[i].r2 << std::endl;
    }
    std::cout << std::string(70, '-') << std::endl;
    
    // Find best method
    size_t bestIndex = 0;
    for (size_t i = 1; i < allMetrics.size(); i++) {
        if (allMetrics[i].rmse < allMetrics[bestIndex].rmse) {
            bestIndex = i;
        }
    }
    
    std::cout << GREEN << "Best method: " << BOLD << methodNames[bestIndex] 
              << RESET << GREEN << " with RMSE of " << std::fixed << std::setprecision(4) 
              << allMetrics[bestIndex].rmse << RESET << std::endl;
}

// Function to expand features with transformations and interactions
Matrix expandFeatures(const std::vector<DataEntry>& data, bool addInteractions = true, 
                     bool addPolynomials = true, bool addLog = true) {
    int baseFeatures = 7; // intercept + 6 original features
    int extraFeatures = 0;
    
    // Count additional features
    if (addPolynomials) extraFeatures += 6; // squared terms
    if (addLog) extraFeatures += 6; // log transforms
    if (addInteractions) extraFeatures += 15; // 6 choose 2 = 15 interaction terms
    
    int totalFeatures = baseFeatures + extraFeatures;
    Matrix X(data.size(), totalFeatures);
    
    // Fill the matrix with data
    for (size_t i = 0; i < data.size(); ++i) {
        const auto& entry = data[i];
        int col = 1;
        
        // Base features (always included)
        X(i+1, col++) = 1.0;  // Intercept
        X(i+1, col++) = entry.MYCT;
        X(i+1, col++) = entry.MMIN;
        X(i+1, col++) = entry.MMAX;
        X(i+1, col++) = entry.CACH;
        X(i+1, col++) = entry.CHMIN;
        X(i+1, col++) = entry.CHMAX;
        
        // Add polynomial terms (squared features)
        if (addPolynomials) {
            X(i+1, col++) = entry.MYCT * entry.MYCT;
            X(i+1, col++) = entry.MMIN * entry.MMIN;
            X(i+1, col++) = entry.MMAX * entry.MMAX;
            X(i+1, col++) = entry.CACH * entry.CACH;
            X(i+1, col++) = entry.CHMIN * entry.CHMIN;
            X(i+1, col++) = entry.CHMAX * entry.CHMAX;
        }
        
        // Add log transforms (add small constant to avoid log(0))
        if (addLog) {
            X(i+1, col++) = std::log(entry.MYCT + 1.0);
            X(i+1, col++) = std::log(entry.MMIN + 1.0);
            X(i+1, col++) = std::log(entry.MMAX + 1.0);
            X(i+1, col++) = std::log(entry.CACH + 1.0);
            X(i+1, col++) = std::log(entry.CHMIN + 1.0);
            X(i+1, col++) = std::log(entry.CHMAX + 1.0);
        }
        
        // Add interaction terms
        if (addInteractions) {
            X(i+1, col++) = entry.MYCT * entry.MMIN;
            X(i+1, col++) = entry.MYCT * entry.MMAX;
            X(i+1, col++) = entry.MYCT * entry.CACH;
            X(i+1, col++) = entry.MYCT * entry.CHMIN;
            X(i+1, col++) = entry.MYCT * entry.CHMAX;
            X(i+1, col++) = entry.MMIN * entry.MMAX;
            X(i+1, col++) = entry.MMIN * entry.CACH;
            X(i+1, col++) = entry.MMIN * entry.CHMIN;
            X(i+1, col++) = entry.MMIN * entry.CHMAX;
            X(i+1, col++) = entry.MMAX * entry.CACH;
            X(i+1, col++) = entry.MMAX * entry.CHMIN;
            X(i+1, col++) = entry.MMAX * entry.CHMAX;
            X(i+1, col++) = entry.CACH * entry.CHMIN;
            X(i+1, col++) = entry.CACH * entry.CHMAX;
            X(i+1, col++) = entry.CHMIN * entry.CHMAX;
        }
    }
    
    return X;
}

// Function to get feature names for the expanded feature matrix
std::vector<std::string> getExpandedFeatureNames(bool addInteractions = true, 
                                               bool addPolynomials = true,
                                               bool addLog = true) {
    std::vector<std::string> featureNames = {"Intercept", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"};
    std::vector<std::string> baseFeatures = {"MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"};
    
    if (addPolynomials) {
        for (const auto& name : baseFeatures) {
            featureNames.push_back(name + "²");
        }
    }
    
    if (addLog) {
        for (const auto& name : baseFeatures) {
            featureNames.push_back("log(" + name + ")");
        }
    }
    
    if (addInteractions) {
        for (size_t i = 0; i < baseFeatures.size(); ++i) {
            for (size_t j = i + 1; j < baseFeatures.size(); ++j) {
                featureNames.push_back(baseFeatures[i] + "*" + baseFeatures[j]);
            }
        }
    }
    
    return featureNames;
}

// Function to detect and remove outliers using Z-score method
std::vector<DataEntry> removeOutliers(const std::vector<DataEntry>& data, double threshold = 3.0) {
    printHeader("OUTLIER DETECTION");
    
    // Calculate mean and std for each feature and the target
    std::vector<double> means(7, 0.0), stdDevs(7, 0.0);
    for (const auto& entry : data) {
        std::vector<double> values = {entry.MYCT, entry.MMIN, entry.MMAX, 
                                      entry.CACH, entry.CHMIN, entry.CHMAX, entry.PRP};
        for (int i = 0; i < 7; ++i) {
            means[i] += values[i];
        }
    }
    
    for (int i = 0; i < 7; ++i) {
        means[i] /= data.size();
    }
    
    for (const auto& entry : data) {
        std::vector<double> values = {entry.MYCT, entry.MMIN, entry.MMAX, 
                                      entry.CACH, entry.CHMIN, entry.CHMAX, entry.PRP};
        for (int i = 0; i < 7; ++i) {
            stdDevs[i] += std::pow(values[i] - means[i], 2);
        }
    }
    
    for (int i = 0; i < 7; ++i) {
        stdDevs[i] = std::sqrt(stdDevs[i] / data.size());
        if (stdDevs[i] == 0) stdDevs[i] = 1.0; // Avoid division by zero
    }
    
    // Identify outliers
    std::vector<bool> isOutlier(data.size(), false);
    int outlierCount = 0;
    
    for (size_t i = 0; i < data.size(); ++i) {
        std::vector<double> values = {data[i].MYCT, data[i].MMIN, data[i].MMAX, 
                                     data[i].CACH, data[i].CHMIN, data[i].CHMAX, data[i].PRP};
        for (int j = 0; j < 7; ++j) {
            double zScore = std::abs((values[j] - means[j]) / stdDevs[j]);
            if (zScore > threshold) {
                isOutlier[i] = true;
                outlierCount++;
                break;
            }
        }
    }
    
    // Copy non-outlier data
    std::vector<DataEntry> cleanedData;
    cleanedData.reserve(data.size() - outlierCount);
    
    for (size_t i = 0; i < data.size(); ++i) {
        if (!isOutlier[i]) {
            cleanedData.push_back(data[i]);
        }
    }
    
    std::cout << GREEN << "[SUCCESS] " << RESET << "Identified " << BOLD << outlierCount 
              << RESET << " outliers (" << std::fixed << std::setprecision(1) 
              << (100.0 * outlierCount / data.size()) << "% of data)" << std::endl;
    std::cout << "Retained " << BOLD << cleanedData.size() << RESET << " data points" << std::endl;
    
    return cleanedData;
}

// Function to find optimal lambda for Tikhonov regularization using k-fold cross-validation
double findOptimalLambda(const Matrix& A, const Vector& b, int kFolds = 5) {
    printHeader("REGULARIZATION PARAMETER TUNING");
    
    std::vector<double> lambdaValues = {0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0};
    std::vector<double> avgRmseValues(lambdaValues.size(), 0.0);
    
    int n = A.GetNumRows();
    int foldSize = n / kFolds;
    
    std::cout << "Performing " << kFolds << "-fold cross-validation to find optimal lambda..." << std::endl;
    
    // Create random indices for shuffling
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i + 1;
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Run cross-validation for each lambda value
    for (size_t l = 0; l < lambdaValues.size(); ++l) {
        double lambda = lambdaValues[l];
        double totalRmse = 0.0;
        
        std::cout << "Testing lambda = " << lambda << " ";
        
        // K-fold cross validation
        for (int fold = 0; fold < kFolds; ++fold) {
            // Create training and validation sets
            int validStartIdx = fold * foldSize + 1;
            int validEndIdx = (fold == kFolds - 1) ? n : (fold + 1) * foldSize;
            
            Matrix trainA(n - (validEndIdx - validStartIdx + 1), A.GetNumCols());
            Vector trainB(n - (validEndIdx - validStartIdx + 1));
            Matrix validA(validEndIdx - validStartIdx + 1, A.GetNumCols());
            Vector validB(validEndIdx - validStartIdx + 1);
            
            // Fill train/valid matrices
            int trainRow = 1, validRow = 1;
            for (int i = 1; i <= n; ++i) {
                int idx = indices[i-1];
                if (i >= validStartIdx && i <= validEndIdx) {
                    // Add to validation set
                    for (int j = 1; j <= A.GetNumCols(); ++j) {
                        validA(validRow, j) = A(idx, j);
                    }
                    validB(validRow) = b(idx);
                    validRow++;
                } else {
                    // Add to training set
                    for (int j = 1; j <= A.GetNumCols(); ++j) {
                        trainA(trainRow, j) = A(idx, j);
                    }
                    trainB(trainRow) = b(idx);
                    trainRow++;
                }
            }
            
            // Train model on training set
            NonSquareLinearSystem solver(trainA, trainB);
            Vector coef = solver.SolveWithTikhonov(lambda);
            
            // Test on validation set
            Vector predictions(validB.getSize());
            for (int i = 1; i <= validB.getSize(); ++i) {
                predictions(i) = 0;
                for (int j = 1; j <= A.GetNumCols(); ++j) {
                    predictions(i) += coef(j) * validA(i, j);
                }
            }
            
            // Calculate RMSE
            double rmse = 0.0;
            for (int i = 1; i <= validB.getSize(); ++i) {
                rmse += std::pow(predictions(i) - validB(i), 2);
            }
            rmse = std::sqrt(rmse / validB.getSize());
            totalRmse += rmse;
            
            std::cout << ".";
            std::cout.flush();
        }
        
        avgRmseValues[l] = totalRmse / kFolds;
        std::cout << " Avg RMSE: " << std::fixed << std::setprecision(4) << avgRmseValues[l] << std::endl;
    }
    
    // Find best lambda
    int bestIdx = 0;
    for (size_t i = 1; i < lambdaValues.size(); ++i) {
        if (avgRmseValues[i] < avgRmseValues[bestIdx]) {
            bestIdx = i;
        }
    }
    
    double bestLambda = lambdaValues[bestIdx];
    std::cout << GREEN << "[SUCCESS] " << RESET << "Optimal lambda: " << BOLD << bestLambda 
              << RESET << " (CV RMSE: " << avgRmseValues[bestIdx] << ")" << std::endl;
    
    return bestLambda;
}

int main() {
    displayVersionInfo();
    auto startTime = std::chrono::high_resolution_clock::now();

    // Load data
    auto data = loadData("machine.data");
    if (data.empty()) {
        std::cerr << RED << "[ERROR] " << RESET << "No data loaded. Exiting." << std::endl;
        return 1;
    }
    
    // Use more conservative outlier removal
    data = removeOutliers(data, 4.0); // Use more lenient Z-score threshold
    
    // Use z-score normalization (mean=0, std=1)
    auto normParams = normalizeData(data, false);
    
    // Split into train/test
    std::vector<DataEntry> trainData, testData;
    trainTestSplit(data, trainData, testData);

    printHeader("MODEL BUILDING");
    std::cout << "Creating linear system from training data..." << std::endl;
    
    // Create standard feature matrix - avoid the complex transformations that cause instability
    Matrix trainX(trainData.size(), 7);  // 7 columns: intercept + 6 features
    Vector trainY(trainData.size());
    
    for (size_t i = 0; i < trainData.size(); ++i) {
        // Fill the feature matrix with the basic features
        trainX(i+1, 1) = 1.0;  // Intercept
        trainX(i+1, 2) = trainData[i].MYCT;
        trainX(i+1, 3) = trainData[i].MMIN;
        trainX(i+1, 4) = trainData[i].MMAX;
        trainX(i+1, 5) = trainData[i].CACH;
        trainX(i+1, 6) = trainData[i].CHMIN;
        trainX(i+1, 7) = trainData[i].CHMAX;
        
        // Target variable
        trainY(i+1) = trainData[i].PRP;
    }

    // Vectors to store results from multiple methods
    std::vector<Vector> allCoefficients;
    std::vector<ModelMetrics> allMetrics;
    std::vector<std::string> methodNames;

    // METHOD 1: Solve using Pseudo-Inverse
    std::cout << "Solving linear system with pseudo-inverse method..." << std::endl;
    NonSquareLinearSystem solver(trainX, trainY);
    Vector coefficients_pseudoinv = solver.SolveWithPseudoInverse();
    std::cout << GREEN << "[SUCCESS] " << RESET << "Pseudo-inverse solution calculated" << std::endl;
    allCoefficients.push_back(coefficients_pseudoinv);
    methodNames.push_back("Pseudo-Inverse");

    // METHOD 2: Solve using Normal Equations + Gaussian elimination
    std::cout << "Solving linear system with normal equations..." << std::endl;
    Matrix AtA = trainX.Transpose() * trainX;
    Vector Atb = trainX.Transpose() * trainY;
    Vector coefficients_gaussian = gaussianElimination(AtA, Atb);
    std::cout << GREEN << "[SUCCESS] " << RESET << "Gaussian elimination solution calculated" << std::endl;
    allCoefficients.push_back(coefficients_gaussian);
    methodNames.push_back("Normal Equations");

    // METHOD 3: Solve using Tikhonov regularization with fixed lambda
    std::cout << "Solving linear system with Tikhonov regularization (lambda=0.1)..." << std::endl;
    Vector coefficients_tikhonov = solver.SolveWithTikhonov(0.1); // Use fixed lambda instead of optimized
    std::cout << GREEN << "[SUCCESS] " << RESET << "Tikhonov regularization solution calculated" << std::endl;
    allCoefficients.push_back(coefficients_tikhonov);
    methodNames.push_back("Tikhonov Regularization");

    // Prepare test data and evaluate all methods
    printHeader("MODEL EVALUATION");
    std::cout << "Evaluating all methods on test set..." << std::endl;
    
    // Create test feature matrix (with same structure as training)
    Matrix testX(testData.size(), 7);
    Vector testY(testData.size());
    
    for (size_t i = 0; i < testData.size(); ++i) {
        testX(i+1, 1) = 1.0;  // Intercept
        testX(i+1, 2) = testData[i].MYCT;
        testX(i+1, 3) = testData[i].MMIN;
        testX(i+1, 4) = testData[i].MMAX;
        testX(i+1, 5) = testData[i].CACH;
        testX(i+1, 6) = testData[i].CHMIN;
        testX(i+1, 7) = testData[i].CHMAX;
        testY(i+1) = testData[i].PRP;
    }
    
    // Evaluate each method
    std::vector<Vector> testPredictions;
    for (size_t method = 0; method < allCoefficients.size(); method++) {
        Vector predictions(testData.size());
        
        // Make predictions
        for (int i = 1; i <= testData.size(); ++i) {
            predictions(i) = 0.0;
            for (int j = 1; j <= testX.GetNumCols(); ++j) {
                predictions(i) += allCoefficients[method](j) * testX(i, j);
            }
        }
        
        testPredictions.push_back(predictions);
        ModelMetrics metrics = calculateMetrics(predictions, testY);
        allMetrics.push_back(metrics);
    }

    std::cout << GREEN << "[SUCCESS] " << RESET << "Model evaluation complete" << std::endl;
    
    // Display comparison of all methods
    compareMethodsTable(allMetrics, methodNames);
    
    // Find best method by lowest RMSE
    size_t bestIndex = 0;
    for (size_t i = 1; i < allMetrics.size(); i++) {
        if (allMetrics[i].rmse < allMetrics[bestIndex].rmse) {
            bestIndex = i;
        }
    }
    
    // Display model summary for the best method
    printModelSummary(allCoefficients[bestIndex], normParams, allMetrics[bestIndex], false, methodNames[bestIndex] + " (Best)");
    
    // Display execution time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    printHeader("EXECUTION SUMMARY");
    std::cout << "Total execution time: " << BOLD << duration / 1000.0 << " seconds" << RESET << std::endl;
    std::cout << "Program completed " << GREEN << "successfully" << RESET << std::endl;
    std::cout << BLUE << std::string(70, '=') << RESET << std::endl;

    return 0;
}