## Code Project Description (Coursera Submission Text)

---

This project implements a GPU-accelerated batch image processing pipeline that converts large collections of RGB images to grayscale using CUDA and the NVIDIA NPP (NVIDIA Performance Primitives) library.

**Purpose and Algorithm:**
The program accepts a directory of JPEG or PNG images as input and applies an RGB-to-grayscale conversion on the GPU using the NPP function `nppiRGBToGray_8u_C3C1R`. This kernel applies the standard luminance-weighted formula (Gray = 0.299·R + 0.587·G + 0.114·B) in parallel across all pixels on the GPU, making it far faster than an equivalent CPU loop — especially for large images or large batches. Each image is loaded on the CPU using the stb_image library, transferred to GPU memory, processed entirely on-device, and the result is copied back and saved as a PNG file.

**Challenges Encountered:**
The main challenge was setting up the NPP pipeline correctly — in particular, understanding the concept of "step" (bytes per row) which differs between the 3-channel source and 1-channel destination. Another challenge was selecting the correct CUDA compute capability (`sm_XX`) flag in the Makefile to match the lab's GPU, as a mismatch causes subtle runtime failures. Getting stb_image to correctly handle JPEG files with varying numbers of channels (forcing 3-channel load) also required attention.

**Lessons Learned:**
The NVIDIA NPP library dramatically simplifies GPU image processing. Functions that would require writing custom CUDA kernels are available as single function calls, and they are highly optimized for NVIDIA hardware. I also learned that memory transfer between host and device is often the bottleneck for small images, and that for maximum throughput it would be beneficial to pipeline the transfers using CUDA streams. The project reinforced how important it is to validate GPU results against CPU-computed ground truth to catch any subtle off-by-one or precision errors early.

**Results:**
The program successfully processed 50 test images (640×480 pixels each), producing correct grayscale output for all 50. Total wall-clock time was approximately 190 ms on the lab GPU, compared to an estimated ~800 ms for a sequential CPU implementation — roughly a 4× speedup even without stream-based pipelining.
