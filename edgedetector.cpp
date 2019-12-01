#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/core/mat.hpp>  
#include <opencv2/imgcodecs.hpp>  
#include <opencv2/imgproc.hpp>  
#include <opencv2/highgui.hpp>  

using namespace std;
using namespace cv;

#define PI 3.141592

void cvOldCanny() {
	//original image
	//********************************************************************//
	/*IplImage* srcImage = cvLoadImage("test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (srcImage == NULL)return 0;

	cvNamedWindow("testimage", CV_WINDOW_AUTOSIZE);
	cvShowImage("testimage", srcImage);

	cvWaitKey(0);
	cvDestroyWindow("testimage");
	cvReleaseImage(&srcImage);*/
	//********************************************************************//

	IplImage* srcImage = cvLoadImage("test5.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (srcImage == NULL) return ;
	IplImage* cannyImage = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_8U, 1);
	IplImage* cannyImage2 = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_8U, 1);
	IplImage* cannyImage3 = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_8U, 1);
	cvNamedWindow("testimage", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Canny", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Canny2", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Canny3", CV_WINDOW_AUTOSIZE);

	cvCanny(srcImage, cannyImage, 50, 100, 3);
	cvCanny(srcImage, cannyImage2, 100, 100, 3);
	cvCanny(srcImage, cannyImage3, 200, 200, 3);
	
	cvShowImage("testimage", srcImage);
	cvShowImage("Canny", cannyImage);
	cvShowImage("Canny2", cannyImage2);
	cvShowImage("Canny3", cannyImage3);
	
	cvWaitKey(0);


	cvDestroyAllWindows();
	cvReleaseImage(&srcImage);
	cvReleaseImage(&cannyImage);
	cvReleaseImage(&cannyImage2);
	cvReleaseImage(&cannyImage3);

}


//기울기 설정
void limit_slope(Mat &img,vector<Vec4i>&lines) {
	for (int i = 0; i < lines.size(); i++)
	{
		Vec4i L = lines[i];
		//caculate slope
		float slope=atan2(L[1] - L[3], L[0] - L[1]) * 180 / PI;
		if (slope<0) slope += 180.0;

		//if (abs(slope) > 160.0 || abs(slope)<60) continue;
		if (slope < 30.0 || slope>170) continue;
		//cout << "slope is : " << slope << "\n";
		line(img, Point(L[0], L[1]), Point(L[2], L[3]),
			Scalar(0, 0, 255), 1, LINE_AA);

	}
	return;
}


//ROI
//in : original img
//out : ROI_image
Mat region_of_interest(Mat &img) {
	Mat mask = Mat::zeros(img.rows,img.cols, CV_8UC1);
	
	Mat ROI_image;
	
	//should modify imperically
	Point corners[1][4];
	corners[0][0] = (Point(0, img.rows - 70));
	corners[0][1]=(Point(img.cols *2/ 5, 70));
	corners[0][2]=(Point((img.cols * 3) / 5, 70));
	corners[0][3]=(Point(img.cols - 70, img.rows - 70));
	const Point *corner_list[1] = { corners[0] };

	int num_points = 4, num_polygons = 1, line_type = 8;

	fillPoly(mask, corner_list, &num_points, num_polygons, Scalar(255, 255, 255), line_type);
	bitwise_and(img, mask, ROI_image);
	//imshow("img", img);
	//imshow("mask", mask);
	//imshow("roi", ROI_image);
	return ROI_image;
}

void draw_lines(Mat& frame, vector<Vec4i>& lines) {
	////검출한 직선을 영상에 그려줍니다.  
	for (int i = 0; i<lines.size(); i++)
	{
		Vec4i L = lines[i];
		line(frame, Point(L[0], L[1]), Point(L[2], L[3]),
			Scalar(0, 0, 255), 1, LINE_AA);
	}
	return;
}

void cannyMat() {
	Mat img_original = imread("test7.jpg", IMREAD_COLOR);
	Mat img_edge, img_gray,img_roi;

	//그레이 스케일 영상으로 변환 한후.  
	cvtColor(img_original, img_gray, COLOR_BGR2GRAY);
	//캐니에지를 이용하여 에지 성분을 검출합니다.  
	Canny(img_gray, img_edge, 90, 270, 3);

	
	img_roi=region_of_interest(img_edge);

	//직선 성분을 검출합니다.  
	vector<Vec4i> lines;
	//HoughLinesP(img_roi, lines, 1, CV_PI / 180, 30, 30, 3);
	HoughLinesP(img_roi, lines, 1, CV_PI / 180, 30, 30, 3);

	limit_slope(img_original, lines);
	//limit_slope(img_edge, lines);

	imshow("draw ", img_edge);
	imshow("img_original", img_original);

	

	waitKey(0);
}

int main() {
	//cannyMat();
	VideoCapture cap("accident_highway.mp4");
	if (!cap.isOpened()) {
		cerr << "Video could not be opened successfully" << "\n";
		return -1;
	}

	cap.set(CAP_PROP_FRAME_WIDTH, 480);
	cap.set(CAP_PROP_FRAME_HEIGHT, 360);

	
	namedWindow("video", 1);

	Mat frame;
	Mat resizeframe;
	Mat img_canny, img_edge, img_gray,img_roi;

	double fps = 0;
	double duration;
	while(char(waitKey(1)!='q' &&cap.isOpened()))
	{
		int64 startTime = getTickCount(); double frequency = getTickFrequency();

		cap.read(frame);
		resize(frame, resizeframe, Size(480, 360), 0, 0, CV_INTER_LINEAR);
		if (frame.empty()) {
			cout << "Video over" << "\n";
			break;
		}

		cvtColor(resizeframe, img_gray, COLOR_BGR2GRAY);
		
		Canny(img_gray, img_edge, 30, 90);

		//ROI setting

		img_roi=region_of_interest(img_edge);
		//직선 성분을 검출합니다.  
		vector<Vec4i> lines;
		HoughLinesP(img_roi, lines, 1, CV_PI / 180, 30, 30, 3);
		
		limit_slope(resizeframe, lines);




		imshow("video", img_edge);
		imshow("video2", resizeframe);
		int64 endTime = getTickCount(); 
		cout << "time : "<<((endTime - startTime) / frequency)*1000<<"msec \n";

		//30ms 정도 대기하도록 해야 동영상이 너무 빨리 재생되지 않음.  
		
	}

	cap.release();
	cv::destroyAllWindows();

	return 0;
}

