#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

Mat computeDFT(Mat image, int &n, int &m) {
    Mat padded;
    m = getOptimalDFTSize( image.rows );
    n = getOptimalDFTSize( image.cols );
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complex;
    merge(planes, 2, complex);
    dft(complex, complex, DFT_COMPLEX_OUTPUT);
    return complex;
}

void shiftDFT(Mat& fImage )
{
  	Mat tmp, q0, q1, q2, q3;
	fImage = fImage(Rect(0, 0, fImage.cols & -2, fImage.rows & -2));

	int cx = fImage.cols/2;
	int cy = fImage.rows/2;

	q0 = fImage(Rect(0, 0, cx, cy));
	q1 = fImage(Rect(cx, 0, cx, cy));
	q2 = fImage(Rect(0, cy, cx, cy));
	q3 = fImage(Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

Mat create_spectrum_magnitude_display(Mat& complexImg, bool rearrange)
{
    Mat planes[2];

    split(complexImg, planes);
    magnitude(planes[0], planes[1], planes[0]);

    Mat mag = (planes[0]).clone();
    mag += Scalar::all(1);
    log(mag, mag);

    if (rearrange)
    {
        shiftDFT(mag);
    }

    normalize(mag, mag, 0, 1, CV_MINMAX);

    return mag;

}

void create_butterworth_lowpass_filter(Mat &dft_Filter, int D, int n, bool inv)
{
	Mat tmp = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);

	Point centre = Point(dft_Filter.rows / 2, dft_Filter.cols / 2);
	double radius;

	for(int i = 0; i < dft_Filter.rows; i++)
	{
		for(int j = 0; j < dft_Filter.cols; j++)
		{
			radius = (double) sqrt(pow((i - centre.x), 2.0) + pow((double) (j - centre.y), 2.0));
			tmp.at<float>(i,j) = (float)
						( 1 / (1 + pow((double) (radius /  D), (double) (2 * n))));
		}
	}

	normalize(tmp, tmp, 0, 1, NORM_MINMAX);

	if(inv)
        tmp = Mat::ones(tmp.size(), CV_32F) - tmp;

    Mat toMerge[] = {tmp, tmp};
	merge(toMerge, 2, dft_Filter);
}

int main( int argc, char** argv )
{
	const char* filename = argc >=2 ? argv[1] : "../data/lena.jpg";
    Mat image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	Mat mag, imgOutput, filter, filterOutput, complexImg, comp, res_mag;
	Mat planes[2];
    if( image.empty()){
        cout << "Error opening image" << endl;
        return -1;
    }
	resize(image, image, Size(image.cols*0.5,image.rows*0.5) );
	int N, M;

	int radius = 30;
	int order = 2;	

	const string originalName = "Input Image";
	const string spectrumMagName = "Result Magnitude";
	const string lowPassName = "Output";
	const string filterName = "Filter Image";

	namedWindow(originalName, CV_WINDOW_AUTOSIZE);
	namedWindow(spectrumMagName, CV_WINDOW_AUTOSIZE);
	namedWindow(lowPassName, CV_WINDOW_AUTOSIZE);
	namedWindow(filterName, CV_WINDOW_AUTOSIZE);
	namedWindow("Magnitude Image", CV_WINDOW_AUTOSIZE);
	
	Mat complexI = computeDFT(image, N, M);

	createTrackbar("Radius", lowPassName, &radius, (min(M, N) / 2));
	createTrackbar("Order", lowPassName, &order, 10);

	while (true){
		complexImg = complexI.clone();
		filter = complexImg.clone();
		comp = complexI.clone();

		create_butterworth_lowpass_filter(filter, radius, order, false);
		
		shiftDFT(complexImg);
		mulSpectrums(complexImg, filter, complexImg, 0);
		shiftDFT(complexImg);

		mag = create_spectrum_magnitude_display(complexImg, true);
		res_mag = create_spectrum_magnitude_display(comp, true);

		idft(complexImg, complexImg);

		split(complexImg, planes);
		normalize(planes[0], imgOutput, 0, 1, CV_MINMAX);

		split(filter, planes);
		normalize(planes[0], filterOutput, 0, 1, CV_MINMAX);

		imshow("Magnitude Image", res_mag);
		imshow(originalName, image);
		imshow(spectrumMagName, mag);
		imshow(lowPassName, imgOutput);
		imshow(filterName, filterOutput);


		if(cvWaitKey(10)==32)
			break;
	}

    return 0;
}
