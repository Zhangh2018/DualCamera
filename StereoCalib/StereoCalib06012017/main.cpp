/********Stereo Calibration for fisheye camera*********/
/********Author: Yang He
*********Date: 06/01/2017
Description: This code is a prototype to explore the stereo calibration
the key function is the stereocalibration, the distortion coefficient is expanded to 
8 values to satisfy the fisheye model's situation****/

#include <string>
#include <stdio.h>
#include <iostream>
#include<fstream>
#include<time.h>
#include<stdio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace std;
using namespace cv;

int main()
{
	ofstream fout("F:\\caliberation_result.txt", ios::app);
	/************************************************************************
	读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
	*************************************************************************/
	cout << "开始提取角点………………" << endl;
	/*****************Initialize Parameters****************/
	int image_count = 20;//**** The number of images for each camera****/ 
	Size boardSize(8, 8);  /****    定标板上每行、列的角点数       ****/
	Size2f squareSize = Size2f(60.f, 60.f);  // Set this to the actual square size
	int successImageNum = 0;        /****   成功提取角点的棋盘图对数量   ****/
	/****  保存检测到的所有角点  ****/
	vector<vector<Point2f> > imagePoints_l; /****  保存检测到的所有角点   ****/
	vector<vector<Point2f> > imagePoints_r; /****  保存检测到的所有角点   ****/
	vector<Mat> image_seq[2]; /****Collect all the calculated sequence*****/ //06/03/2017
	int nimages = 0; /****累加总的角点数*****/

	for (int i = 0; i != image_count; i++)
	{
		cout << "Left Frame #" << i + 1 << "..." << endl;
		cout << "Right Frame #" << i + 1 << "..." << endl;
		/******Load the Image******/
		string imageFileName;
		std::stringstream StrStm;
		StrStm << i + 1;
		StrStm >> imageFileName;
		imageFileName += ".bmp";
		Mat image_left = imread("F:\\yanghe\\fisheye\\MDG001\\NO2\\MDG2_L" + imageFileName);
		if (image_left.empty())
		{
			cout << "Cannot open the left images" << endl;
			break;
		}
		Mat image_right = imread("F:\\yanghe\\fisheye\\MDG001\\NO2\\MDG2_r" + imageFileName);
		if (image_left.empty())
		{
			cout << "Cannot open the right images" << endl;
			break;
		}
		Mat imageGray_left, imageGray_right;
		cvtColor(image_left, imageGray_left, COLOR_BGR2GRAY);
		cvtColor(image_right, imageGray_right, COLOR_BGR2GRAY);

		/*******Find the Corner*******/
		bool found_l = false, found_r = false;
		vector<Point2f>corners_l, corners_r; //a tempary container for the individual frame
		//Left Camera
		found_l = findChessboardCorners(imageGray_left, boardSize, corners_l,
			CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
		if (!found_l)
		{
			cout << "can not find chessboard corners!\n";
			continue;
			exit(1);
		}
		else
		{
			cout << "find the left chessboard corners!\n";
			cornerSubPix(imageGray_left, corners_l, Size(11, 11),
				Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
		}

		//Right Camera
		found_r = findChessboardCorners(imageGray_right, boardSize, corners_r,
			CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
		if (!found_r)
		{
			cout << "can not find chessboard corners!\n";
			continue;
			exit(1);
		}
		else
		{
			cout << "find the right chessboard corners!\n";
			cornerSubPix(imageGray_right, corners_r, Size(11, 11),
				Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
		}

		//put the calculated images in the sequence
		bool bgetImageSeq = true;
		if (bgetImageSeq)
		{
			image_seq[0].push_back(imageGray_left);
			image_seq[1].push_back(imageGray_right);
		}

		//Draw and label the corners
		if (found_l && found_r)
		{
			drawChessboardCorners(image_left, boardSize, corners_l, found_l);
			drawChessboardCorners(image_right, boardSize, corners_r, found_r);

			imwrite("F:\\yanghe\\left" + imageFileName, image_left);
			imwrite("F:\\yanghe\\right" + imageFileName, image_right);
			cout << "Find the #"<<i+1<<" pairs" << endl;
			//Group the corners
			imagePoints_l.push_back(corners_l);
			imagePoints_r.push_back(corners_r);
			successImageNum = successImageNum + 1;
		}
	}

	//cout << "角点提取完成！\n";

	//if (nimages < 20){ cout << "Not enough" << endl; return -1; }

	//vector<vector<Point2f> > imagePoints[2] = { imagePoints_l, imagePoints_r };

	/*******INFORMATION OF THE CORNERS ON THE CHESSBOAR******/	
	vector<vector<Point3f> > object_Points;  /****  保存定标板上角点的三维坐标   ****/
	
	for (int t = 0; t < successImageNum; t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0; i < boardSize.height; i++)
		{
			for (int j = 0; j < boardSize.width; j++)
			{
				/* 假设定标板放在世界坐标系中z=0的平面上 */
				Point3d tempPoint;
				tempPoint.x = i * squareSize.width;
				tempPoint.y = j * squareSize.height;
				tempPoint.z = 0;
				tempPointSet.push_back(tempPoint);//For the individual image
			}
		}
		object_Points.push_back(tempPointSet);//For all images
	}

	/************************************************************************
	摄像机定标
	*************************************************************************/
	cout << "开始定标………………" << endl;
	cout << "Running stereo calibration ..." << endl;
	//Initialize the input parameters
	Size imageSize(1280, 960);

	/*************************************************************************
	摄像机内参数矩阵
	***********************************************************************/
	//LEFT CAMERA
	Mat intrinsic_matrix_left = Mat(3, 3, CV_64F);
	intrinsic_matrix_left.at<double>(0, 0) = 1314;
	intrinsic_matrix_left.at<double>(0, 2) = 2313;
	intrinsic_matrix_left.at<double>(1, 1) = 1314;
	intrinsic_matrix_left.at<double>(1, 2) = 1631;
	intrinsic_matrix_left.at<double>(0, 1) = 0;
	intrinsic_matrix_left.at<double>(1, 0) = 0;
	intrinsic_matrix_left.at<double>(2, 0) = 0;
	intrinsic_matrix_left.at<double>(2, 1) = 0;
	intrinsic_matrix_left.at<double>(2, 2) = 1;

	Mat distortion_coeffs_left = Mat(8, 1, CV_64F);
	distortion_coeffs_left.at<double>(0) = -0.465;//k1
	distortion_coeffs_left.at<double>(1) = 0.015;//k2
	distortion_coeffs_left.at<double>(2) = 0.0;//p1
	distortion_coeffs_left.at<double>(3) = 0.0;//p2
	distortion_coeffs_left.at<double>(4) = -0.010;//k3
	distortion_coeffs_left.at<double>(5) = 0.002;//k4
	distortion_coeffs_left.at<double>(6) = 0.0;//k5
	distortion_coeffs_left.at<double>(7) = 0.0;//k6

	//RIGHT CAMERA
	Mat intrinsic_matrix_right = Mat(3, 3, CV_64F);
	intrinsic_matrix_right.at<double>(0, 0) = 1314;
	intrinsic_matrix_right.at<double>(0, 2) = 2313;
	intrinsic_matrix_right.at<double>(1, 1) = 1314;
	intrinsic_matrix_right.at<double>(1, 2) = 1631;
	intrinsic_matrix_right.at<double>(0, 1) = 0;
	intrinsic_matrix_right.at<double>(1, 0) = 0;
	intrinsic_matrix_right.at<double>(2, 0) = 0;
	intrinsic_matrix_right.at<double>(2, 1) = 0;
	intrinsic_matrix_right.at<double>(2, 2) = 1;


	Mat distortion_coeffs_right = Mat(8, 1, CV_64F);
	distortion_coeffs_right.at<double>(0) = -0.465;//k1
	distortion_coeffs_right.at<double>(1) = 0.015;//k2
	distortion_coeffs_right.at<double>(2) = 0.0;//p1
	distortion_coeffs_right.at<double>(3) = 0.0;//p2
	distortion_coeffs_right.at<double>(4) = -0.010;//k3
	distortion_coeffs_right.at<double>(5) = 0.002;//k4
	distortion_coeffs_right.at<double>(6) = 0.0;//k5
	distortion_coeffs_right.at<double>(7) = 0.0;//k6

	/*****************************************************************************/

	Mat R, T;
	Mat E, F;
	double rms =
		stereoCalibrate(object_Points,
		imagePoints_l,
		imagePoints_r,
		intrinsic_matrix_left, distortion_coeffs_left,
		intrinsic_matrix_right, distortion_coeffs_right,
		imageSize, R, T, E, F,
		CV_CALIB_ZERO_TANGENT_DIST,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5)
		);
	cout << "done with RMS error=" << rms << endl;

	/************************************************************************
	Assess the accuracy by the epipolar line constraints
	**************************************************************************/
		double err_avr = 0;
		int points_total = 0;// (# of points in the individual iamge) * (# of images)
		vector<Point3f> lines[2]; //for the epipolar lines
		 
		
		for (int i = 0; i < successImageNum; i++)
		{
			int npt = (int)imagePoints_l[i].size(); //# of points in the individual image
			//Mat image_point_temp[2]; //To store the matched points in two cameras 
		
			//image_point_temp[0] = Mat(imagePoints_l[i]);
			vector<Point2f> & point_left_temp = imagePoints_l[i];
			undistortPoints(point_left_temp, point_left_temp,
		 			intrinsic_matrix_left, distortion_coeffs_left,
					Mat(), intrinsic_matrix_left);
			computeCorrespondEpilines(point_left_temp, 0 + 1, F, lines[0]);
		 
			vector<Point2f> & point_right_temp = imagePoints_r[i];
			undistortPoints(point_right_temp, point_right_temp,
		 			intrinsic_matrix_right, distortion_coeffs_right,
		 			Mat(), intrinsic_matrix_right);
			computeCorrespondEpilines(point_right_temp, 1 + 1, F, lines[1]);
		 
				for (int j = 0; j < npt; j++)
				{
					double err = fabs(point_left_temp[j].x * lines[1][j].x +
						point_left_temp[j].y * lines[1][j].y + lines[1][j].z) +
						fabs(point_right_temp[j].x * lines[0][j].x +
						point_right_temp[j].y * lines[0][j].y + lines[0][j].z);
					err_avr += err;
				}
				points_total += npt;
		 	}
		 
	err_avr = err_avr / points_total;
	cout << "average epipolar line error = " << err_avr << endl;
	/*****************************************************************************/
	
	/************************************************************************
	保存定标结果
	*************************************************************************/
	cout << "开始保存定标结果………………" << endl;

// 	time_t t = time(0);
// 	char time_temp[64];
// 	strftime(time_temp, sizeof(time_temp), "%Y/%m/%d %X %A", localtime(&t));
// 	fout << time_temp << endl;

	fout << "Left Intrinsic Matrix:" << endl;
	fout << intrinsic_matrix_left << endl;
	fout << "Left DIstortion Coefficient" << endl;
	fout << distortion_coeffs_left << endl;

	fout << "Right Intrinsic Matrix:" << endl;
	fout << intrinsic_matrix_right << endl;
	fout << "Right DIstortion Coefficient" << endl;
	fout << distortion_coeffs_right << endl;

	fout << "Rotation Vector:" << endl;
	fout << R << endl;
	fout << "Translation Vector:" << endl;
	fout << T << endl;
	fout << "RMS Error:" << endl;
	fout << rms << endl;
	fout << "Epipolar error:" << endl;
	fout << err_avr << endl;
	cout << "完成保存" << endl;
	fout << endl;
	fout.close();
	
/***********************************************************************/

	/************************************************************************
	RECTIFICATION 06/02/2017
	*************************************************************************/
	Mat R1, R2, P1, P2, Q,
		map11,map12,
		map21,map22;
	Rect validRoi[2];
	
	stereoRectify(intrinsic_matrix_left, distortion_coeffs_left,
	 		intrinsic_matrix_right, distortion_coeffs_right,
	 		imageSize, R, T, R1, R2, P1, P2, Q,
	 		CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

	initUndistortRectifyMap(intrinsic_matrix_left, distortion_coeffs_left, R1, P1, imageSize, CV_16SC2, map11, map12);
	initUndistortRectifyMap(intrinsic_matrix_right, distortion_coeffs_right, R2, P2, imageSize, CV_16SC2, map21, map22);
	
	Mat imageRect_left, imageRect_right;
	cvtColor(image_seq[0][0], imageRect_left, CV_GRAY2BGR);
	cvtColor(image_seq[1][0], imageRect_right, CV_GRAY2BGR);
	
	remap(imageRect_left, imageRect_left, map11, map12, INTER_LINEAR);
	remap(imageRect_right, imageRect_right, map21, map22, INTER_LINEAR);

	imwrite("F:\\yanghe\\rect1.bmp", imageRect_left);
	imwrite("F:\\yanghe\\rectr.bmp", imageRect_right);
	/************************************************************************
	保存定标结果
	*************************************************************************/
	bool bsaveYMLResult = 0;
	if (bsaveYMLResult)
	{
		FileStorage fs("intrinsics.yml", FileStorage::WRITE);
		if (fs.isOpened())
		{
			fs << "Translation" << T << "Rotation" << R;
			fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
			fs.release();
		}
		else
			cout << "Error: can not save the Result\n";
	}

	return 0;
}