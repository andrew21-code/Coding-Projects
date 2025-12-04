#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <unordered_set>
#include <chrono>
#include <unordered_map>
#include <omp.h>
#include <limits>
#include <algorithm>

using namespace std;

// g++-14 -std=c++17 -O3 -fopenmp parallel.cpp -o parallel

// Function to load MNIST dataset
void load_MNIST(const char* images_file, const char* labels_file,
                vector<vector<float>>& images, vector<int>& labels) {
    int rows = 70000, cols = 784;

    ifstream file(images_file);
    if (!file) {
        cerr << "Error opening images file!" << endl;
        return;
    }

    ifstream file2(labels_file);
    if (!file2) {
        cerr << "Error opening labels file!" << endl;
        return;
    }

    images.resize(rows, vector<float>(cols));
    labels.resize(rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file >> images[i][j];
        }
        file2 >> labels[i];
    }
    file.close();
    file2.close();
}

// Function to add new images, using gaussian noise
pair<vector<int>, vector<vector<float>>> add_images(const vector<vector<float>>& images, const vector<int>& labels, int k) {
    vector<vector<float>> noisy_images = images;  // Copy the original images to manipulate
    float noise_level = 0.1f;
    vector<int> noisy_labels = labels;

    random_device rd;  
    mt19937 gen(rd()); 
    normal_distribution<float> dis(0, 15);

    for (int i = 0; i < k; i++) {
        int index = rand() % images.size();
        vector<float> chosen_image = images[index];
        
        // Add Random noise to each pixel
        for (float &pixel : chosen_image) {
            float noise = dis(gen);
            pixel = max(0.0f, min(255.0f, pixel + noise));  // Ensure pixel values remain in [0,255]
        }

        noisy_images.push_back(chosen_image);
        noisy_labels.push_back(labels[index]);
    }

    return {noisy_labels, noisy_images};
}

class KMeans {
private:
    int max_iterations;
    int k;

public:
    KMeans(int k) : k(k) {}

    // Function to calculate Euclidean distance
    float distance(const vector<float>& a, const vector<float>& b) {
        float sum = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }

    // K-means++ initialization
    vector<vector<float>> kmeanspp(const vector<vector<float>>& images, int k) {
        vector<vector<float>> centroids;
        unordered_set<int> chosen_indexes;

        srand(time(0));

        // Step 1: Choose a random centroid
        int first_centroid_idx = rand() % images.size();
        centroids.push_back(images[first_centroid_idx]);
        chosen_indexes.insert(first_centroid_idx);

        // Step 2: Compute the minimum distance from each point to the closest centroid
        vector<float> min_distances(images.size(), numeric_limits<float>::max());

        while (centroids.size() < k) {
            float total_distance = 0.0;

            // Update the minimum distance for each point
            #pragma omp parallel for reduction(+:total_distance)
            for (int j = 0; j < images.size(); j++) {
                if (chosen_indexes.find(j) != chosen_indexes.end()) continue;

                float dist = distance(images[j], centroids.back());
                if (dist < min_distances[j]) {
                    min_distances[j] = dist;
                }
                total_distance += min_distances[j];
            }

            // Step 3: Choose the next centroid proportional to the squared distance
            float target = static_cast<float>(rand()) / RAND_MAX * total_distance;
            float cumulative_distance = 0.0;
            int next_centroid_idx = -1;
            

            for (int j = 0; j < images.size(); j++) {
                if (chosen_indexes.find(j) != chosen_indexes.end()) continue;

                cumulative_distance += min_distances[j];
                if (cumulative_distance >= target) {
                    next_centroid_idx = j;
                    break;
                }
            }

            if (chosen_indexes.find(next_centroid_idx) == chosen_indexes.end()) {
                centroids.push_back(images[next_centroid_idx]);
                chosen_indexes.insert(next_centroid_idx);
            }
        }

        return centroids;
    }


    // Find nearest centroid
    int nearestCentroid(const vector<vector<float>>& centroids, const vector<float>& image) {

        float min_dist = numeric_limits<float>::max();
        int best_cluster = -1;

        
        for (int i = 0; i < centroids.size(); i++) {
            float dis = distance(centroids[i], image);
            if (dis < min_dist) {
                best_cluster = i;
                min_dist = dis;
            }
        }

        return best_cluster;
    }

    // Mini-batch k-means clustering algorithm
    pair<vector<int>, vector<vector<float>>> MiniBatch(const vector<vector<float>>& images, const vector<int>& labels,
                                                       int k, int batch_size) {
        int max_iterations = 200;

        vector<vector<float>> centroids = kmeanspp(images, k); // Initialize centroids

        vector<int> cluster_labels(images.size(), 0); // Initialize cluster labels
        vector<int> counts(k, 0); // Initialize counts to count the elements in each cluster

        // Perform mini-batch k-means
        for (int i = 0; i < max_iterations; i++) {
            random_device rd;
            mt19937 gen(rd());

            // Choose a random sample of batch_size unique indexes of images
            unordered_set<int> chosen_set;
            vector<int> chosen_indexes;

            while (chosen_indexes.size() < batch_size) {
                int random_index = gen() % images.size();
                if (chosen_set.find(random_index) == chosen_set.end()) {
                    chosen_set.insert(random_index);
                    chosen_indexes.push_back(random_index);
                }
            }

            // Find nearest centroid for every batch image
            vector<int> nearest_clusters(chosen_indexes.size(), -1);

            #pragma omp parallel for schedule(guided)
            for (int idx = 0; idx < chosen_indexes.size(); idx++) {
                int index = chosen_indexes[idx];
                nearest_clusters[idx] = nearestCentroid(centroids, images[index]);
                cluster_labels[index] = nearest_clusters[idx];
            }


            // Update centroids based on the assigned clusters
            #pragma omp parallel for schedule(guided) 
            for (int idx = 0; idx < chosen_indexes.size(); idx++) {
                int index = chosen_indexes[idx];
                int nearest_cluster = nearest_clusters[idx];

                #pragma omp atomic
                counts[nearest_cluster]++;

                float learning_rate = 1.0 / counts[nearest_cluster];

                #pragma omp critical
                {
                    for (int j = 0; j < centroids[nearest_cluster].size(); j++) {
                        centroids[nearest_cluster][j] = (1 - learning_rate) * centroids[nearest_cluster][j] +
                                                        learning_rate * images[index][j];
                    }
                }
            }

            
        }

        // Perform one last scan of the dataset to assign every object to the closest cluster
        #pragma omp parallel for schedule(guided) 
        for (int i = 0; i < images.size(); i++) {
            cluster_labels[i] = nearestCentroid(centroids, images[i]);
        }

        return {cluster_labels, centroids};
    }

    double normalized_mutual_information(const vector<int>& true_labels, const vector<int>& cluster_labels) {
        unordered_map<int, int> cluster_count, label_count;
        unordered_map<int, unordered_map<int, int>> joint_count;

        int N = true_labels.size();
        for (int i = 0; i < N; i++) {
            cluster_count[cluster_labels[i]]++;
            label_count[true_labels[i]]++;
            joint_count[true_labels[i]][cluster_labels[i]]++;
        }

        double H_C = 0.0;
        double H_L = 0.0;
        double MI = 0.0;

        for (auto& c : cluster_count) {
            double p_c = (double)c.second / N;
            H_C -= p_c * log2(p_c);
        }
        for (auto& l : label_count) {
            double p_l = (double)l.second / N;
            H_L -= p_l * log2(p_l);
        }
        for (auto& l : joint_count) {
            for (auto& c : l.second) {
                double p_cl = (double)c.second / N;
                MI += p_cl * log2(p_cl / (((double)label_count[l.first] / N) * ((double)cluster_count[c.first] / N)));
            }
        }
        return MI / ((H_C + H_L) / 2.0);
    }

    // Function to compute the total error
    float compute_error(const vector<vector<float>>& centroids, const vector<int>& cluster_labels, const vector<vector<float>>& images) {
        float error = 0.0;
        for (int i = 0; i < images.size(); i++) {
            int cluster = cluster_labels[i];
            float dist = distance(images[i], centroids[cluster]);
            error += dist * dist;
        }
        return error;
    }

};

int main() {
    vector<vector<float>> images;
    vector<int> labels;
    

    load_MNIST("mnist-images.txt", "mnist-labels.txt", images, labels);

    if (images.empty()) {
        cerr << "Failed to load images." << endl;
        return 1;
    }

    // int value = 10000; 
    // auto augmented_data = add_images(images, labels, value);
    // labels = augmented_data.first;
    // images = augmented_data.second;

    omp_set_num_threads(20); // Set OpenMP threads

    int k = 50;

    int b = 5000;

    KMeans kmeans(k);

    auto start_time_5 = chrono::high_resolution_clock::now();
    pair<vector<int>, vector<vector<float>>> result = kmeans.MiniBatch(images, labels, k, b);
    auto end_time_5 = chrono::high_resolution_clock::now();
    
    chrono::duration<double> elapsed5 = end_time_5 - start_time_5;
    cout << "Mini-Batch algorithm time: " << elapsed5.count() << " s" << endl;

    vector<int> cluster_labels = result.first;
    vector<vector<float>> centroids = result.second;

    float nmi = kmeans.normalized_mutual_information(labels, cluster_labels);
    cout << "Normalized Mutual Information (NMI): " << nmi << endl;

    float err = kmeans.compute_error(centroids, cluster_labels, images);
    cout << "Error: " << err << endl;

    
    return 0;
}
