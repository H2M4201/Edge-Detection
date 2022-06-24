// 19120241_BT00.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv2\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>
#include <cmath>
#define pi 3.14159265359
using namespace std;
using namespace cv;

int RGB_to_Gray(cv::Mat &src, cv::Mat &dest);
int Grad_Operation(cv::Mat &img, vector<vector<int>> filter, int x, int y);
int Detect_by_Sobel (cv::Mat &src, cv::Mat &dest);
int Detect_by_Prewitt(cv::Mat &src, cv::Mat &dest);
int Detect_by_Laplace(cv::Mat &src, cv::Mat &dest);
int Detect_by_Canny(cv::Mat &src, cv::Mat &dest, double upper_threshold, double lower_threshold);
void Gaussian_blur(cv::Mat &img);
void Non_maximum_supression(cv::Mat &img, int x, int y, int angle);
bool is_largest(int a, int b, int c);
bool is_connect_to_strong_edge(cv::Mat &img, int x, int y, int upper_bound);
vector<vector<bool>> keep_or_supress(cv::Mat &img, double upper_bound, double lower_bound);

int main(int argc, char** argv)
{
	cv::String keys =
		"{@image | /path/to/image | duong dan den tap tin}"
		"{-t type| sobel prewitt laplace canny | thuat toan dung de xu ly}"
		"{-c color | rgb gray | loai mau cua anh dau vao (default: color) }"
		"{help h || in thong tin huong dan}";


	cv::CommandLineParser parser(argc, argv, keys);
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	cv::String image_path = parser.get<cv::String>(0); 
	if (image_path == "/path/to/image") {
		cout << "Error: Vui long nhap duong dan tap tin\n";
		parser.printMessage();
		return 0;
	}
	cv::Mat image;
	image = imread(image_path, IMREAD_COLOR);
	if (!image.data)
	{
		cout << "Khong the mo anh" << std::endl;
		return -1;
	}
	namedWindow("Display window", WINDOW_AUTOSIZE); // (3)
	cv::imshow("Display window", image); // (4)
	bool hasTypeParam = parser.has("type");
	bool hasColorParam = parser.has("color");
	cv::String type_arg;
	if (hasTypeParam) {
		type_arg = parser.get<cv::String>("type");
	}
	else {
		cout << "Error: Vui long chon thuan toan (sobel|prewitt|laplace|canny) !\n";
		parser.printMessage();
		return -1;
	}
	cv::Mat GrayImage = Mat(image.rows, image.cols, CV_8UC1);
	if (hasColorParam) {
		cv::String type_arg = parser.get<cv::String>("color");
		if (type_arg == "rgb") {
			RGB_to_Gray(image, GrayImage);
		}
		else if (type_arg == "gray") {
			GrayImage = image;
		}
		else {
			RGB_to_Gray(image, GrayImage);
		}
	}
	else {
		RGB_to_Gray(image, GrayImage);
	}
	Mat output = Mat(image.rows - 2, image.cols - 2, CV_8UC1);
	if (type_arg == "sobel") {
		Detect_by_Sobel(GrayImage, output);
	} else if (type_arg == "prewitt") {
		Detect_by_Prewitt(GrayImage, output);
		} else if (type_arg == "laplace") {
			Detect_by_Laplace(GrayImage, output);
			} else if (type_arg == "canny") {
				Detect_by_Canny(GrayImage, output, 50, 150);
				} else {
					cout << "Error: Vui long chon mot thuan toan hop le (sobel|prewitt|laplace|canny) !\n";
					return -1;
				}
	waitKey(0);
	return 0;
}

int RGB_to_Gray(cv::Mat &src, cv::Mat &dest)
{
	if (src.data == NULL)
		return 0;

	int c = src.cols, r = src.rows;
	int src_channel = src.channels();
	int dest_channel = dest.channels();


	for (int i = 0; i < c; i++)
	{
		uchar* src_pRow = src.ptr<uchar>(i);
		uchar* dest_pRow = dest.ptr<uchar>(i);
		for (int j = 0; j < r; j++, src_pRow += src_channel, dest_pRow += dest_channel)
		{
			uchar B = src_pRow[0];
			uchar G = src_pRow[1];
			uchar R = src_pRow[2];
			uchar gray = (uchar)(0.3*R + 0.59*G + 0.11*B);
			dest_pRow[0] = gray;
		}
	}
	return 1;
}

int Grad_Operation(cv::Mat &img, vector<vector<int>> filter, int x,int y)
{
	return (int)(img.at<uchar>(x - 1, y - 1)) * filter[0][0]
		+ (int)(img.at<uchar>(x - 1, y)) * filter[0][1]
		+ (int)(img.at<uchar>(x - 1, y + 1)) * filter[0][2]
		+ (int)(img.at<uchar>(x, y - 1)) * filter[1][0]
		+ (int)(img.at<uchar>(x, y)) * filter[1][1]
		+ (int)(img.at<uchar>(x, y + 1)) * filter[1][2]
		+ (int)(img.at<uchar>(x + 1, y - 1)) * filter[2][0]
		+ (int)(img.at<uchar>(x + 1, y)) * filter[2][1]
		+ (int)(img.at<uchar>(x + 1, y + 1)) * filter[2][2];
}

int Detect_by_Sobel(cv::Mat &src, cv::Mat &dest)
{
	if (src.data == NULL)
		return 0;

	int c = src.cols, r = src.rows;
	dest = Mat(c, r, CV_8UC1);

	if (c < 3 || r < 3)
	{
		cout << "Kich thuoc anh khong phu hop !";
		return 0;
	}
	else
	{
		vector<vector<int>> x_filter = { { -1, 0, 1 },{ -2, 0, 2 },{ -1, 0, 1 } };
		vector<vector<int>> y_filter = { { 1, 2, 1 },{ 0, 0, 0 },{ -1, -2, -1 } };
		Mat x_grad = Mat(r, c, CV_8UC1);
		Mat y_grad = Mat(r, c, CV_8UC1);

		for (int i = 1; i < r - 1; i++)
		{
			for (int j = 1; j < c - 1; j++)
			{
				int dx = (int)(Grad_Operation(src, x_filter, i, j));
				int dy = (int)(Grad_Operation(src, y_filter, i, j));
				x_grad.at<uchar>(i, j) = dx;
				y_grad.at<uchar>(i, j) = dy;
				dest.at<uchar>(i, j) = (int)(sqrt(dx*dx + dy*dy));
			}
		}

		Gaussian_blur(dest);

		namedWindow("Display x-grad", WINDOW_AUTOSIZE); // (3)
		cv::imshow("Display x-grad", x_grad); // (4)

		namedWindow("Display y-grad", WINDOW_AUTOSIZE); // (3)
		cv::imshow("Display y-grad", y_grad); // (4)

		namedWindow("Display grad", WINDOW_AUTOSIZE); // (3)
		cv::imshow("Display grad", dest); // (4)

		return 1;
	}
}

int Detect_by_Prewitt(cv::Mat &src, cv::Mat &dest)
{
	if (src.data == NULL)
		return 0;

	int c = src.cols, r = src.rows;
	dest = Mat(c, r, CV_8UC1);

	if (c < 3 || r < 3)
	{
		cout << "Kich thuoc anh khong phu hop !";
		return 0;
	}
	else
	{
		vector<vector<int>> x_filter = { { -1, 0, 1 },{ -1, 0, 1 },{ -1, 0, 1 } };
		vector<vector<int>> y_filter = { { 1, 1, 1 },{ 0, 0, 0 },{ -1, -1, -1 } };
		Mat x_grad = Mat(r, c, CV_8UC1);
		Mat y_grad = Mat(r, c, CV_8UC1);

		for (int i = 1; i < r - 1; i++)
		{
			for (int j = 1; j < c - 1; j++)
			{
				x_grad.at<uchar>(i, j) = Grad_Operation(src, x_filter, i, j);
				y_grad.at<uchar>(i, j) = Grad_Operation(src, y_filter, i, j);
				dest.at<uchar>(i, j) = (int)(sqrt(
					Grad_Operation(src, x_filter, i, j) *  Grad_Operation(src, x_filter, i, j)
					+ Grad_Operation(src, y_filter, i, j)* Grad_Operation(src, y_filter, i, j)));
			}
		}

		Gaussian_blur(dest);

		namedWindow("Display x-grad", WINDOW_AUTOSIZE); // (3)
		imshow("Display x-grad", x_grad); // (4)

		namedWindow("Display y-grad", WINDOW_AUTOSIZE); // (3)
		imshow("Display y-grad", y_grad); // (4)

		namedWindow("Display grad", WINDOW_AUTOSIZE); // (3)
		imshow("Display grad", dest); // (4)

		return 1;
	}
}

int Detect_by_Laplace(cv::Mat &src, cv::Mat &dest)

{
	if (src.data == NULL)
		return 0;

	int c = src.cols, r = src.rows;
	dest = Mat(c, r, CV_8UC1);

	if (c < 3 || r < 3)
	{
		cout << "Kich thuoc anh khong phu hop !";
		return 0;
	}
	else
	{
		Gaussian_blur(src);
		vector<vector<int>> x_filter = { { -1, -1, -1 },{ -1, 8, -1 },{ -1, -1, -1 } };
		vector<vector<int>> y_filter = { { -1, -1, -1 },{ -1, 8, -1 },{ -1, -1, -1 } };
		Mat x_grad = Mat(r, c, CV_8UC1);
		Mat y_grad = Mat(r, c, CV_8UC1);

		for (int i = 1; i < r - 1; i++)
		{
			for (int j = 1; j < c - 1; j++)
			{
				x_grad.at<uchar>(i, j) = Grad_Operation(src, x_filter, i, j);
				y_grad.at<uchar>(i, j) = Grad_Operation(src, y_filter, i, j);
				dest.at<uchar>(i, j) = (int)(sqrt(
					Grad_Operation(src, x_filter, i, j) *  Grad_Operation(src, x_filter, i, j)
					+ Grad_Operation(src, y_filter, i, j)* Grad_Operation(src, y_filter, i, j)));
			}
		}

		namedWindow("Display x-grad", WINDOW_AUTOSIZE); // (3)
		imshow("Display x-grad", x_grad); // (4)

		namedWindow("Display y-grad", WINDOW_AUTOSIZE); // (3)
		imshow("Display y-grad", y_grad); // (4)

		namedWindow("Display grad", WINDOW_AUTOSIZE); // (3)
		imshow("Display grad", dest); // (4)

		return 1;
	}
}

int Detect_by_Canny(cv::Mat &src, cv::Mat &dest, double upper_threshold, double lower_threshold)
{
	if (src.data == NULL)
		return 0;

	int c = src.cols, r = src.rows;
	dest = Mat(c, r, CV_8UC1);

	if (c < 3 || r < 3)
	{
		cout << "Kich thuoc anh khong phu hop !";
		return 0;
	}
	else
	{
		// noise processing
		Gaussian_blur(src);
		vector<vector<int>> x_filter = { { -1, 0, 1 },{ -1, 0, 1 },{ -1, 0, 1 } };
		vector<vector<int>> y_filter = { { 1, 1, 1 },{ 0, 0, 0 },{ -1, -1, -1 } };

		// gradient matrix and direction detecting
		Mat x_grad = Mat(r, c, CV_8UC1);
		Mat y_grad = Mat(r, c, CV_8UC1);
		vector<vector<int>> angle;

		for (int i = 1; i < r - 1; i++)
		{
			vector<int> tmp;
			for (int j = 1; j < c - 1; j++)
			{
				// counting gradient using Prewitt kernel
				double dx = Grad_Operation(src, x_filter, i, j);
				double dy = Grad_Operation(src, y_filter, i, j);
				x_grad.at<uchar>(i, j) = (uchar)(dx);
				y_grad.at<uchar>(i, j) = (uchar)(dy);
				dest.at<uchar>(i, j) = (int)(sqrt(dx*dx + dy*dy));

				// detecting angle
				if (dx == 0)
					dx = 0.000001; // avoid dividing zero
				double a = 180.0 * atan2(1.0 * dy, 1.0 * dx) / pi;

				// round up the angle
				if (a < 0) a += 180;
				if (a < 22.5) a = 0;
				else if (a < 67.5) a = 45;
				else if (a < 112.5) a = 90;
				else if (a < 157.5) a = 135;
				else a = 180.0;
				tmp.push_back((int)(a));
			}
			angle.push_back(tmp);
		}

		// lower bound thresholding
		for (int i = 1; i < c - 1; i++)
		{
			for (int j = 1; j < r - 1; j++)
			{
				int a = angle[i - 1][j - 1];
				Non_maximum_supression(dest, i, j, a);
			}
		}

		// double thresholding
		vector<vector<bool>> edge_or_noise = keep_or_supress(dest, upper_threshold, lower_threshold);
		for (int i = 1; i < c - 1; i++)
		{
			for (int j = 1; j < r - 1; j++)
			{
				if (edge_or_noise[i - 1][j - 1] == false)
					dest.at<uchar>(i, j) = (uchar)(0);
			}
		}


		namedWindow("Display x-grad", WINDOW_AUTOSIZE); // (3)
		imshow("Display x-grad", x_grad); // (4)

		namedWindow("Display y-grad", WINDOW_AUTOSIZE); // (3)
		imshow("Display y-grad", y_grad); // (4)

		namedWindow("Display grad", WINDOW_AUTOSIZE); // (3)
		imshow("Display grad", dest); // (4)

		return 1;
	}
}

void Gaussian_blur(cv::Mat &img)
{
	vector<vector<int>> gauss_filter = { { 1, 2, 1 }, // Gauss kernel
										 { 2, 4, 2 }, 
										 { 1, 2, 1 } }; 
	int c = img.cols, r = img.rows;
		for (int i = 1; i < c - 1; i++)
		{
			for (int j = 1; j < r - 1; j++)
				img.at<uchar>(i, j) = Grad_Operation(img, gauss_filter, i, j) / 16;
		}
}

void Non_maximum_supression(cv::Mat &img, int x, int y, int angle)
{
	if (angle == 0 || angle == 180)
	{
		int t1 = int(img.at<uchar>(x, y));
		int t2 = int(img.at<uchar>(x, y - 1));
		int t3 = int(img.at<uchar>(x, y + 1));
		if (!is_largest(t1, t2, t3))
			img.at<uchar>(x, y) = (uchar)(0);
	}
	if (angle == 45)
	{
		int t1 = int(img.at<uchar>(x, y));
		int t2 = int(img.at<uchar>(x + 1, y - 1));
		int t3 = int(img.at<uchar>(x - 1, y + 1));
		if (!is_largest(t1, t2, t3))
			img.at<uchar>(x, y) = (uchar)(0);
	}
	if (angle == 90)
	{
		int t1 = int(img.at<uchar>(x, y));
		int t2 = int(img.at<uchar>(x - 1, y));
		int t3 = int(img.at<uchar>(x + 1, y));
		if (!is_largest(t1, t2, t3))
			img.at<uchar>(x, y) = (uchar)(0);
	}
	if (angle == 135)
	{
		int t1 = int(img.at<uchar>(x, y));
		int t2 = int(img.at<uchar>(x - 1, y - 1));
		int t3 = int(img.at<uchar>(x + 1, y + 1));
		if (!is_largest(t1, t2, t3))
			img.at<uchar>(x, y) = (uchar)(0);
	}
}

bool is_largest(int a, int b, int c) //check if a is the largest of all 3
{
	int tmp = max(a, b);
	if (tmp == b)
		return false;
	else
	{
		tmp = max(a, c);
		if (tmp == c)
			return false;
		return true;
	}
}

bool is_connect_to_strong_edge(cv::Mat &img, int x, int y, double upper_bound)
{
	int c = img.cols, r = img.rows;
	if (x == 0 || x == c - 1 || y == 0 || y == c - 1)
		return false;
	else
	{
		int count = 0; //count the number of adjacent strong edge pixel
		if ((int)(img.at<uchar>(x - 1, y - 1)) > upper_bound) count++;
		if ((int)(img.at<uchar>(x - 1, y)) > upper_bound) count++;
		if ((int)(img.at<uchar>(x - 1, y + 1)) > upper_bound) count++;
		if ((int)(img.at<uchar>(x, y - 1)) > upper_bound) count++;
		if ((int)(img.at<uchar>(x - 1, y + 1)) > upper_bound) count++;
		if ((int)(img.at<uchar>(x + 1, y - 1)) > upper_bound) count++;
		if ((int)(img.at<uchar>(x + 1, y)) > upper_bound) count++;
		if ((int)(img.at<uchar>(x + 1, y + 1)) > upper_bound) count++;
		if (count > 0) return true;
		return false;
	}
}

vector<vector<bool>> keep_or_supress(cv::Mat &img, double upper_bound, double lower_bound)
{
	// true -> strong edge -> keep that pixel
	// false -> not and edge -> supress
	// Note: In this function, the weak edge will be categorized as noise or strong edge
	vector<vector<bool>> res;
	int c = img.cols, r = img.rows;
	for (int i = 1; i < r - 1; i++)
	{
		vector<bool> tmp;
		for (int j = 1; j < c - 1; j++)
		{
			int v = (int)(img.at<uchar>(i, j));
			if (v > upper_bound)
				tmp.push_back(true);
			else
			{
				if (v < lower_bound)
					tmp.push_back(false);
				else
				{
					bool noise_or_edge = is_connect_to_strong_edge(img, i, j, upper_bound);
					if (noise_or_edge == true)
						tmp.push_back(true);
					else
						tmp.push_back(false);
				}
			}
		}
		res.push_back(tmp);
	}
	return res;
}
