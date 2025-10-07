/* 
  Special thanks to my Mom, Dad and ChatGPT for making this code possible. 
  This is an utility library that allows to store image both in the form of bitmap and
  compressed version using DFT (Discrete Fourier Transform) coefficients.
  Please treat this code as read-only external dependency - it's knowledge is not even 
  needed to pass the task.
*/

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>

#define COLORS_COUNT 3 // red, green, blue

typedef void (CompressFun)(const uint32_t valuesCount, const int accuracy, 
    const uint8_t*, float*, float*);
typedef void (DecompressFun)(const uint32_t valuesCount, const int accuracy, 
    uint8_t*, const float*, const float*);

std::chrono::high_resolution_clock::time_point timeNow() {
  return std::chrono::high_resolution_clock::now();
}

typedef std::chrono::milliseconds millis;
typedef std::chrono::microseconds micro;

#pragma pack(push, 1)
struct BMPHeader {
    char header_field[2];
    uint32_t file_size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
    uint32_t header_size;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bits_per_pixel;
    uint32_t compression;
    uint32_t image_size;
    uint32_t r_pixels_per_meter;
    uint32_t y_pixels_per_meter;
    uint32_t colors_used;
    uint32_t important_colors;
};
#pragma pack(pop)

struct BMP {
  // RGB represents bitmap in a form of 3 dimensional array.
  // RGB[colorIndex] keeps bitmap of a single color.
  // RBG[colorIndex][row][column] returns value of the specific color at (row, column) position.
  uint8_t *RGB = nullptr;
  // Xreal[colorIndex][row] keeps coefficient for the specific color and row. 
  // Xreal and Ximag length depends on <accuracy> value.
  float *Xreal = nullptr, *Ximag = nullptr;
  int accuracy;

  ~BMP() {
      delete[] RGB;
      delete[] Xreal;
      delete[] Ximag;
  }

  BMPHeader header;

  void read(std::string filename) {
    std::ifstream bmpFile(filename, std::ios::binary);

    bmpFile.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    if (header.header_field[0] != 'B' || header.header_field[1] != 'M' || 
        header.bits_per_pixel != 24 || header.compression != 0) {
      std::cerr << "Invalid file format. Must be 24-bit uncompressed BMP" << std::endl;
      exit(1);
    }

    size_t width = header.width, height = header.height;
    
    RGB = new uint8_t[COLORS_COUNT * width * height];
    
    int rowPadding = (4 - ((width * 3) % 4)) % 4;

    for (int r = height - 1; r >= 0; r--) { // BMP files are stored upside-down
        for (uint c = 0; c < width; c++) {
            for (int colorIndex=0; colorIndex<COLORS_COUNT; colorIndex++) {
              uint8_t value;
              bmpFile.read(reinterpret_cast<char*>(&value), sizeof(value));       
              RGB[width * (colorIndex * height + r) + c] = value;
            }
        }

        bmpFile.seekg(rowPadding, std::ios::cur); // Skip row padding
    }
  }

  void write(std::string filename) {
    std::ofstream bmpFile(filename, std::ios::binary);

    bmpFile.write(reinterpret_cast<char*>(&header), sizeof(header));
    size_t width = header.width, height = header.height;
    
    int rowPadding = (4 - ((width * header.bits_per_pixel / 8) % 4)) % 4; // Pad each row to a multiple of 4 bytes
    
    for (int r = height - 1; r >= 0; r--) { // BMP files are stored upside-down
        for (int c = width - 1; c >= 0; c--) {
            for (int colorIndex=0; colorIndex<COLORS_COUNT; colorIndex++) {
              uint8_t value = RGB[width * (colorIndex * height + r) + c];
              bmpFile.write(reinterpret_cast<char*>(&value), sizeof(value));    
            }
	      }
        uint8_t padding_value = 0;
        bmpFile.write(reinterpret_cast<char*>(&padding_value), sizeof(uint8_t) * rowPadding);
    }
    bmpFile.close();
  }

  float compress(CompressFun *fun, size_t accuracy) {
      size_t width = header.width, height = header.height;

      float sizeRatio = accuracy * 2.0 / width;
      float typeRatio = sizeof(float) / (float) sizeof(uint8_t);
      fprintf(stderr, "Compression ratio %.2f\n", sizeRatio * typeRatio);
      micro totalTime{0};

      Xreal = new float[COLORS_COUNT * height * accuracy]();
      Ximag = new float[COLORS_COUNT * height * accuracy]();

      for (int colorIndex=0; colorIndex<COLORS_COUNT; colorIndex++) {

          for (uint j=0; j<height; j++) {
            auto startTime = timeNow();
            fun(
              width, accuracy,
              RGB + width * (colorIndex * height + j), 
              Xreal + (colorIndex * height + j) * accuracy, 
              Ximag + (colorIndex * height + j )* accuracy
            );            
            totalTime += std::chrono::duration_cast<micro>(timeNow() - startTime);
          }
      }

      this->accuracy = accuracy;

      delete[] RGB;
      RGB = nullptr;

      return std::chrono::duration_cast<millis>(totalTime).count() / 1000.0;
  }

  float decompress(DecompressFun *fun) {
      size_t width = header.width, height = header.height;
      micro totalTime{0};

      RGB = new uint8_t[COLORS_COUNT * width * height];

      for (int colorIndex=0; colorIndex<COLORS_COUNT; colorIndex++) {
          for (uint j=0; j<height; j++) {
            auto startTime = timeNow();
            fun(
              width, accuracy,
              RGB + width * (colorIndex * height + j), 
              Xreal + (colorIndex * height + j) * accuracy, 
              Ximag + (colorIndex * height + j )* accuracy
            );
          totalTime += std::chrono::duration_cast<micro>(timeNow() - startTime);
          }
      }

      delete[] Xreal;
      delete[] Ximag;
      Xreal = Ximag = nullptr;

      return std::chrono::duration_cast<millis>(totalTime).count() / 1000.0;
  }
};
