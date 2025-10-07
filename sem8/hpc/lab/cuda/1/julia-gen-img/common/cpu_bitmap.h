#ifndef __CPU_BITMAP_H__
#define __CPU_BITMAP_H__

#include <fstream>

struct CPUBitmap {
	unsigned char *pixels;
	int           x,y;
	void          *dataBlock;
	void (*bitmapExit)(void *);
	
	CPUBitmap(int width, int height, void *d = NULL) {
		pixels = new unsigned char[width * height * 4];
		x = width;
		y = height;
		dataBlock = d;
	}
	
	~CPUBitmap() {
		delete [] pixels;
	}
	
	unsigned char* get_ptr(void) const { return pixels; }
	long image_size(void) const { return x * y * 4; }
	
	// dumps bitmap in PPM (no alpha channel)
	void dump_ppm(const char *fname) {
		std::ofstream out(fname);
		out << "P3" << std::endl;
		out << x << " " << y << " 255" << std::endl;
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++)
				out << (int) pixels[4 * (i * y + j) + 0] << " "
				    << (int) pixels[4 * (i * y + j) + 1] << " "
				    << (int) pixels[4 * (i * y + j) + 2] << " ";
			out << std::endl;
		}
	}
};

#endif // __CPU_BITMAP_H__
		
