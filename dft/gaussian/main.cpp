#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

void shift(Mat magI) {
 
    // crop if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
 
    int cx = magI.cols/2;
    int cy = magI.rows/2;
 
    Mat q0(magI, Rect(0, 0, cx, cy));
    Mat q1(magI, Rect(cx, 0, cx, cy));
    Mat q2(magI, Rect(0, cy, cx, cy));
    Mat q3(magI, Rect(cx, cy, cx, cy));
 
    Mat tmp;                           
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void updateResult(Mat complex)
{
    Mat work;
    idft(complex, work);
    //  dft(complex, work, DFT_INVERSE + DFT_SCALE);
    Mat planes[] = {Mat::zeros(complex.size(), CV_32F), Mat::zeros(complex.size(), CV_32F)};
    split(work, planes);                // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
 
    magnitude(planes[0], planes[1], work);    // === sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
    normalize(work, work, 0, 1, NORM_MINMAX);
    imshow("result", work);
}
 
void updateMag(Mat complex, string winName )
{
 
    Mat magI;
    Mat planes[] = {Mat::zeros(complex.size(), CV_32F), Mat::zeros(complex.size(), CV_32F)};
    split(complex, planes);                // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
 
    magnitude(planes[0], planes[1], magI);    // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
 
    // switch to logarithmic scale: log(1 + magnitude)
    magI += Scalar::all(1);
    log(magI, magI);
 
    shift(magI);
    normalize(magI, magI, 1, 0, NORM_INF); // Transform the matrix with float values into a
                                              // viewable image form (float between values 0 and 1).
    imshow(winName, magI);
}
  
Mat computeDFT(Mat image) {
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( image.rows );
    int n = getOptimalDFTSize( image.cols ); // on the border add zero values
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complex;
    merge(planes, 2, complex);         // Add to the expanded another plane with zeros
    dft(complex, complex, DFT_COMPLEX_OUTPUT);  // furier transform
    return complex;
}

double pixelDistance(double u, double v)
{
    return cv::sqrt(u*u + v*v);
}

double gaussianCoeff(double u, double v, double d0)
{
    double d = pixelDistance(u, v);
    return 1.0 - cv::exp((-d*d) / (2*d0*d0));
}

void createGaussianFilter(Mat &dft_Filter, double cutoffInPixels, bool inv)
{
    Mat ghpf = Mat(dft_Filter.rows, dft_Filter.cols, CV_32F);
    
    cv::Point center(dft_Filter.cols / 2, dft_Filter.rows / 2);

    for(int u = 0; u < ghpf.rows; u++)
    {
        for(int v = 0; v < ghpf.cols; v++)
        {
            ghpf.at<float>(u, v) = gaussianCoeff(u - center.y, v - center.x, cutoffInPixels);
        }
    }

    // transform mask to range 0..1
    normalize(ghpf, ghpf, 0, 1, NORM_MINMAX);
    
    if(inv)
        ghpf = Mat::ones(ghpf.size(), CV_32F) - ghpf;
    
    Mat toMerge[] = {ghpf, ghpf};
	merge(toMerge, 2, dft_Filter);
}

int cut_off = 10;
 
int main( int argc, char** argv )
{ 
    const char* filename = argc >=2 ? argv[1] : "../data/lena.jpg";
    Mat mag, imgOutput, filter, filterOutput, complexImg, comp;
    Mat image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if( image.empty()){
        cout << "Error opening image" << endl;
        return -1;
    }
    resize(image, image, Size(image.cols*0.5,image.rows*0.5) );
    namedWindow( "Orginal image", CV_WINDOW_AUTOSIZE  );
    imshow( "Orginal image", image );
 
    Mat complex = computeDFT(image);
 
    namedWindow( "result", CV_WINDOW_AUTOSIZE );
    createTrackbar( "Cut off", "result", &cut_off, 255, 0 );

    const string filterName = "Filter";
    const string spect = "Magnitude Image";
	const string spectrumName = "Result Magnitude";
    
    while(true){
        complexImg = complex.clone();
		filter = complexImg.clone();
        comp = complex.clone();

        createGaussianFilter(filter, cut_off, true);
        
        shift(complexImg);
        mulSpectrums(complexImg, filter, complexImg, 0);
        shift(complexImg);
        shift(filter);
        // shift(comp);

        updateMag(comp,spect);
        updateMag(filter,filterName);
        updateMag(complexImg,spectrumName);
        updateResult(complexImg);

        if(cvWaitKey(10)==32)
			break;
    }

    return 0;
}
