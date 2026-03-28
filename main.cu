// main.cu
// Batch Image Grayscale Converter using CUDA NPP
// Converts a folder of RGB images to grayscale using GPU acceleration.
// Follows Google C++ Style Guide.

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <chrono>

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>

// stb_image for loading/saving images (header-only)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helper macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                << " - " << cudaGetErrorString(err) << "\n";                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define NPP_CHECK(call)                                                        \
  do {                                                                         \
    NppStatus status = (call);                                                 \
    if (status != NPP_SUCCESS) {                                               \
      std::cerr << "NPP error at " << __FILE__ << ":" << __LINE__             \
                << " - code " << status << "\n";                              \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// ProcessImage: loads one RGB image, converts to grayscale on GPU, saves out
// ---------------------------------------------------------------------------
bool ProcessImage(const std::string& input_path,
                  const std::string& output_path) {
  // --- Load image from disk (CPU) ---
  int width = 0, height = 0, channels = 0;
  unsigned char* h_rgb =
      stbi_load(input_path.c_str(), &width, &height, &channels, 3);
  if (!h_rgb) {
    std::cerr << "  [WARN] Could not load: " << input_path << "\n";
    return false;
  }

  const int kChannels = 3;
  const size_t rgb_bytes  = static_cast<size_t>(width * height * kChannels);
  const size_t gray_bytes = static_cast<size_t>(width * height);

  // --- Allocate GPU memory ---
  Npp8u* d_rgb  = nullptr;
  Npp8u* d_gray = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rgb),  rgb_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gray), gray_bytes));

  // --- Copy RGB image to GPU ---
  CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice));

  // --- NPP RGB -> Grayscale conversion ---
  NppiSize roi = {width, height};
  int src_step  = width * kChannels;  // bytes per row (source)
  int dst_step  = width;              // bytes per row (destination)

  NPP_CHECK(nppiRGBToGray_8u_C3C1R(
      d_rgb,  src_step,
      d_gray, dst_step,
      roi));

  // --- Copy result back to CPU ---
  unsigned char* h_gray = new unsigned char[gray_bytes];
  CUDA_CHECK(cudaMemcpy(h_gray, d_gray, gray_bytes, cudaMemcpyDeviceToHost));

  // --- Save grayscale image ---
  stbi_write_png(output_path.c_str(), width, height, 1, h_gray, width);

  // --- Cleanup ---
  stbi_image_free(h_rgb);
  delete[] h_gray;
  CUDA_CHECK(cudaFree(d_rgb));
  CUDA_CHECK(cudaFree(d_gray));

  return true;
}

// ---------------------------------------------------------------------------
// CollectImages: returns all .jpg/.jpeg/.png files under a directory
// ---------------------------------------------------------------------------
std::vector<std::string> CollectImages(const std::string& dir) {
  std::vector<std::string> paths;
  for (const auto& entry : fs::recursive_directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    std::string ext = entry.path().extension().string();
    // Convert extension to lowercase for comparison
    for (char& c : ext) c = static_cast<char>(std::tolower(c));
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
      paths.push_back(entry.path().string());
    }
  }
  return paths;
}

// ---------------------------------------------------------------------------
// PrintUsage
// ---------------------------------------------------------------------------
void PrintUsage(const char* program_name) {
  std::cout << "Usage: " << program_name
            << " --input <input_dir> --output <output_dir> [--log <log_file>]\n"
            << "\n"
            << "  --input   Path to directory containing input images\n"
            << "  --output  Path to directory where grayscale images are saved\n"
            << "  --log     (Optional) Path to log file (default: stdout)\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  std::string input_dir;
  std::string output_dir;
  std::string log_file;

  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--input" && i + 1 < argc) {
      input_dir = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "--log" && i + 1 < argc) {
      log_file = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return EXIT_SUCCESS;
    }
  }

  if (input_dir.empty() || output_dir.empty()) {
    std::cerr << "Error: --input and --output are required.\n\n";
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  // Set up logging
  std::ofstream log_stream;
  std::ostream* log = &std::cout;
  if (!log_file.empty()) {
    log_stream.open(log_file);
    if (!log_stream.is_open()) {
      std::cerr << "Warning: could not open log file, using stdout.\n";
    } else {
      log = &log_stream;
    }
  }

  // Create output directory if needed
  fs::create_directories(output_dir);

  // Print CUDA device info
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  *log << "=== CUDA Batch Grayscale Converter ===\n"
       << "GPU: " << prop.name << "\n"
       << "CUDA devices found: " << device_count << "\n\n";

  // Collect images
  std::vector<std::string> images = CollectImages(input_dir);
  if (images.empty()) {
    std::cerr << "No images found in: " << input_dir << "\n";
    return EXIT_FAILURE;
  }
  *log << "Found " << images.size() << " image(s) in: " << input_dir << "\n\n";

  // Process each image
  int success_count = 0;
  int fail_count    = 0;
  auto t_start = std::chrono::high_resolution_clock::now();

  for (const auto& img_path : images) {
    fs::path src(img_path);
    std::string out_name = src.stem().string() + "_gray.png";
    std::string out_path = (fs::path(output_dir) / out_name).string();

    *log << "Processing: " << src.filename().string() << " -> " << out_name;
    *log << std::flush;

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok  = ProcessImage(img_path, out_path);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (ok) {
      *log << "  [OK]  " << ms << " ms\n";
      ++success_count;
    } else {
      *log << "  [FAIL]\n";
      ++fail_count;
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double total_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

  *log << "\n=== Summary ===\n"
       << "Total images processed : " << success_count + fail_count << "\n"
       << "Successful             : " << success_count << "\n"
       << "Failed                 : " << fail_count << "\n"
       << "Total wall-clock time  : " << total_ms << " ms\n"
       << "Output directory       : " << output_dir << "\n";

  return EXIT_SUCCESS;
}
