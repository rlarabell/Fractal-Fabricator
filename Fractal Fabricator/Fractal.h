#pragma once

extern "C"
{
	__declspec(dllexport) void get_pixel_data(int* output_data, int width, int height);
}