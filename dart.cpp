/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - dart.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <cmath>

#define PI 3.14159265

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, char* imageName );
void Sobel(cv::Mat &input, cv::Mat &xOutput, cv::Mat &yOutput, cv::Mat &magnitude, cv::Mat &direction);
cv::Mat Convolve(cv::Mat &input, cv::Mat &paddedInput, cv::Mat_<double> kernel, int kernelRadiusX, int kernelRadiusY);
void HoughLines(int **&houghSpace, cv::Mat edges, cv::Mat direction, int threshold, int mind, int maxd);
int **malloc2dArray(int dim1, int dim2);


/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

// Ground Truth dart board locations
int groundTruths[16][3][4] = {{{437,5,169,202},{0,0,0,0},{0,0,0,0}},
															{{185,123,216,214},{0,0,0,0},{0,0,0,0}},
															{{98,95,96,92},{0,0,0,0},{0,0,0,0}},
															{{321,146,72,77},{0,0,0,0},{0,0,0,0}},
															{{174,87,227,231},{0,0,0,0},{0,0,0,0}},
															{{430,137,109,119},{0,0,0,0},{0,0,0,0}},
															{{210,117,67,66},{0,0,0,0},{0,0,0,0}},
															{{247,164,161,165},{0,0,0,0},{0,0,0,0}},
															{{840,219,125,123},{68,252,59,94},{0,0,0,0}},
															{{190,42,259,249},{0,0,0,0},{0,0,0,0}},
															{{91,102,102,114},{582,124,59,88},{917,148,37,65}},
															{{176,101,60,69},{443,118,56,67},{0,0,0,0}},
															{{160,75,60,148},{0,0,0,0},{0,0,0,0}},
															{{270,118,141,142},{0,0,0,0},{0,0,0,0}},
															{{119,97,137,134},{984,90,134,134},{0,0,0,0}},
															{{152,51,137,146},{0,0,0,0},{0,0,0,0}}};



/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	char *imageName = strdup(argv[1]);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect darts and Display Result
	detectAndDisplay( frame, imageName );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}



/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, char* imageName )
{
	std::vector<Rect> darts;
	Mat frame_gray;
	int **centres = malloc2dArray(3,2);

	// Prepare Image by turning it into Grayscale
	cvtColor( frame, frame_gray, CV_BGR2GRAY );

	// Perform Sobel edge detection to obtain magnitude and direction of gradient images.
	Mat xOutput, yOutput, magnitude, direction, edges;
	Sobel(frame_gray, xOutput, yOutput, magnitude, direction);

	// Perform Canny edge detection to obtain cleaner edge image.
	Canny(frame_gray, edges, 115, 380, 3);
	imwrite("edges.png", edges);

	// Perform intersecting lines Hough Transform on edge image to find possible dartboard centres.
	int **houghSpace = malloc2dArray(edges.size[0], edges.size[1]);
	HoughLines(houghSpace, edges, direction, 215, 1, 50);

	//Use houghSpace to calculate number and centre of dartboards.
	int numCentres = 0;
	for (int a = 0; a < edges.size[0]; a++) {
		for (int b = 0; b < edges.size[1]; b++) {
			if (houghSpace[a][b] > 215) {
				circle(frame, Point(b, a), 5, cvScalar(0, 0, 255), 2, 8, 0);
				centres[numCentres][0] = a;
				centres[numCentres][1] = b;
				numCentres++;
				a += 10;
				b += 10;
			}	
		}	
	}

	//Print number of dartboards found.
	printf("DARTBOARDS FOUND: %d\n", numCentres);

	// Perform Viola-Jones Object Detection
	equalizeHist( frame_gray, frame_gray );
	cascade.detectMultiScale( frame_gray, darts, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

	//Calculate the true number of dartboards from the ground truth.
	int truthNum = 0;
	if (!strcmp(imageName, "dart0.jpg")) truthNum = 0;
	if (!strcmp(imageName, "dart1.jpg")) truthNum = 1;
	if (!strcmp(imageName, "dart2.jpg")) truthNum = 2;
	if (!strcmp(imageName, "dart3.jpg")) truthNum = 3;
	if (!strcmp(imageName, "dart4.jpg")) truthNum = 4;
	if (!strcmp(imageName, "dart5.jpg")) truthNum = 5;
	if (!strcmp(imageName, "dart6.jpg")) truthNum = 6;
	if (!strcmp(imageName, "dart7.jpg")) truthNum = 7;
	if (!strcmp(imageName, "dart8.jpg")) truthNum = 8;
	if (!strcmp(imageName, "dart9.jpg")) truthNum = 9;
	if (!strcmp(imageName, "dart10.jpg")) truthNum = 10;
	if (!strcmp(imageName, "dart11.jpg")) truthNum = 11;
	if (!strcmp(imageName, "dart12.jpg")) truthNum = 12;
	if (!strcmp(imageName, "dart13.jpg")) truthNum = 13;
	if (!strcmp(imageName, "dart14.jpg")) truthNum = 14;
	if (!strcmp(imageName, "dart15.jpg")) truthNum = 15;

	double numDarts = 0;
	for (int i = 0; i < 3; i++) {
		if (groundTruths[truthNum][i][0] != 0) numDarts++;
	}

  // Draw box around dartboards found
	int foundBoards = 0;
	int** detectedBoards = malloc2dArray(3, 4);
	while (foundBoards < numCentres) {
		for( int i = 0; i < darts.size(); i++ ) {
			if (centres[foundBoards][0] > darts[i].y+20 && centres[foundBoards][1] > darts[i].x+20 &&
				centres[foundBoards][0] < darts[i].y + darts[i].height-20 && centres[foundBoards][1] < darts[i].x + darts[i].width-20) {
				detectedBoards[foundBoards][0] = darts[i].x;
				detectedBoards[foundBoards][1] = darts[i].y;
				detectedBoards[foundBoards][2] = darts[i].width;
				detectedBoards[foundBoards][3] = darts[i].height;
				foundBoards++;
				rectangle(frame, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar( 0, 255, 0 ), 2);
			}	
		}	
	}

	// Calculate F1 score from ground truth.
	double truePos = 0;
	double falsePos = 0;

	for ( int i = 0; i < numCentres; i++ ) {
		bool isDart = false;
		for (int j = 0; j < numDarts; j++) {
			bool xOverlap = abs(centres[i][1] - (groundTruths[truthNum][j][0] + groundTruths[truthNum][j][2]/2)) < 20;
			bool yOverlap = abs(centres[i][0] - (groundTruths[truthNum][j][1] + groundTruths[truthNum][j][3]/2)) < 20;
			bool widthOverlap = abs(detectedBoards[i][2] - groundTruths[truthNum][j][2]) < 100;
			bool heightOverlap = abs(detectedBoards[i][3] - groundTruths[truthNum][j][3]) < 100;
			if (xOverlap && yOverlap && widthOverlap && heightOverlap) isDart = true;
			if (xOverlap && yOverlap) isDart = true;
		}
		if (isDart) truePos++;
		else falsePos++;
	}

	double recall, precision, score;

	precision = truePos / (truePos+falsePos);
	recall = truePos / numDarts;
	score = 2 * (recall*precision)/(recall+precision);

	printf("F1-SCORE: %.3f\n", score);
}


/** @function Sobel */
void Sobel(cv::Mat &input, cv::Mat &xOutput, cv::Mat &yOutput, cv::Mat &magnitude, cv::Mat &direction)
{
	// Intialise the Sobel output images.
	xOutput.create(input.size(), input.type());
	yOutput.create(input.size(), input.type());
	magnitude.create(input.size(), CV_32F);
	direction.create(input.size(), CV_32F);

	// Create the kernels
	cv::Mat_<double> xKernel(3,3);
	xKernel << -1, 0, 1, -2, 0, 2, -1, 0, 1;

	cv::Mat_<double> yKernel(3,3);
	yKernel << -1, -2, -1, 0, 0, 0, 1, 2, 1;

	// Create a padded version of the input
	// to prevent border effects
	int xKernelRadiusX = ( xKernel.size[0] - 1 ) / 2;
	int xKernelRadiusY = ( xKernel.size[1] - 1 ) / 2;

	int yKernelRadiusX = ( yKernel.size[0] - 1 ) / 2;
	int yKernelRadiusY = ( yKernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		xKernelRadiusX, xKernelRadiusX, xKernelRadiusY, xKernelRadiusY,
		cv::BORDER_REPLICATE );

	// Use Sobel kernels to obtain X and Y derivative images
	xOutput = Convolve(input, paddedInput, xKernel, xKernelRadiusX, xKernelRadiusY);
	yOutput = Convolve(input, paddedInput, yKernel, yKernelRadiusX, yKernelRadiusY);

	// Calculate magnitude of gradient image from X and Y derivative images.
	cv::magnitude(xOutput, yOutput, magnitude);

	// Calculate direction of gradient image from X and Y derivative images.
	for (int i=0; i < direction.size[0]; i++) {
		for (int j=0; j < direction.size[1]; j++) {
			direction.at<float>(i, j) = atan2(xOutput.at<float>(i, j), yOutput.at<float>(i, j));
		}
	}
}


/** @function Convolve */
cv::Mat Convolve(cv::Mat &input, cv::Mat &paddedInput, cv::Mat_<double> kernel, int kernelRadiusX, int kernelRadiusY)
{
	// Create output image.
	Mat output;
	output.create(input.size(), CV_32F);

	// Perform Convolution.
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// Find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// Get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// Do the multiplication
					sum += imageval * kernalval;
				}
			}
			// Set the output value as the sum of the convolution
			output.at<float>(i, j) = (float) sum;
		}
	}
	return output;
}


/** @function houghLines */
void HoughLines(int **&houghSpace, cv::Mat edges, cv::Mat direction, int threshold, int mind, int maxd)
{
	Mat houghOutput;
	houghOutput.create(edges.size(), CV_8U);
	houghOutput = Scalar::all(0);

	// Perform Hough Transform for intersecting lines.
	for (int x = 0; x < edges.size[0]; x++) {
		for (int y = 0; y < edges.size[1]; y++) {
			uchar pixel = edges.at<uchar>(x,y);
			if (pixel != 0) {
				// Use direction of gradient image to calculate line gradient.
				float alpha = direction.at<float>(x, y);
				for (int dist = mind; dist < maxd; dist++) {
					// Small room for error in direction angle.
					for (float t = -2; t < 2; t++) {
						float theta = alpha + PI/2 + (t * PI/180);
						int a = x + dist*cos(theta);
						int b = y + dist*sin(theta);
						if (a >= 0 && b >= 0 && a < edges.size[0] && b < edges.size[1]) {
							//Vote for possible centre of intersecting lines.
							houghSpace[a][b]++;
							houghOutput.at<uchar>(a, b)++;
						}
					}
				}
			}
		}
	}
	imwrite( "hough.jpg", houghOutput );
}


/** @function malloc2dArray */
int **malloc2dArray(int dim1, int dim2)
{
		// Allocate memory for 2D array.
    int i, j;
    int **array = (int **) malloc(dim1 * sizeof(int *));
    for (i = 0; i < dim1; i++) {
        array[i] = (int *) malloc(dim2 * sizeof(int));
	}
	for (int x = 0; x < dim1; x++) {
		for (int y = 0; y < dim2; y++) {
			array[x][y] = 0;
		}
	}
    return array;
}
