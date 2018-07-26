#ifndef _STRUCTURE_MEASURE_H
#define _STRUCTURE_MEASURE_H

#pragma once
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;

#define MAX_MAT_SIZE 2048
static double eps = 2.2204e-16;

void checkRGBMap(Mat& Map){
	if (Map.channels() != 1){
		cvtColor(Map, Map, CV_RGB2GRAY);
	}
	Map.convertTo(Map, CV_32FC1);
}

Mat& getStaticMat(){
	static Mat base_mat(MAX_MAT_SIZE, MAX_MAT_SIZE, CV_32FC1, 1.0);
	return base_mat;
}

Mat& getStaticInreaseRowVector(){
	static Mat base_vec(1, MAX_MAT_SIZE, CV_32FC1, 0.0);
	if (base_vec.at<float>(0, 0) <= 0){
		for (int i = 0; i < MAX_MAT_SIZE; i++){
			base_vec.at<float>(0, i) = (float)(i + 1);
		}
	}
	return base_vec;
}
/*
void init(){
	//init normlize function 
	Mat normlize_init_mat(1, 1, CV_32FC1, 1.0);
	normalize(normlize_init_mat, normlize_init_mat, 0, 1, NORM_MINMAX);
	//init static vector
	Mat A = getStaticMat();
	Mat B = getStaticInreaseRowVector();
}*/

//search centroid
void centroid(Mat& GT, int& x, int& y){
	
	double total = sum(GT).val[0];
	if (total <= 0 || (GT.rows <= 2 && GT.cols <= 2)){
		y = GT.rows / 2;
		x = GT.cols / 2;
	}
	
	Mat mat_for_x = (getStaticMat())(Rect(0, 0, GT.rows, 1))*GT;
	
	Mat tmp = (getStaticInreaseRowVector())(Rect(0, 0, mat_for_x.cols, 1));
	double sum_x = sum(mat_for_x.mul(tmp)).val[0];
	x = (int)round(sum_x / total) - 1;

	Mat mat_for_y = GT*(getStaticMat())(Rect(0, 0, 1, GT.cols));
	double sum_y = sum((mat_for_y.t()).mul((getStaticInreaseRowVector())(Rect(0, 0, mat_for_y.rows, 1)))).val[0];
	y = (int)round(sum_y / total) - 1;

}


void DivideGT(Mat& GT, int x, int y, Mat& LT, Mat&  RT, Mat&  LB, Mat&  RB, float& w1, float& w2, float& w3, float& w4){
	/*
	% LT - left top;
	% RT - right top;
	% LB - left bottom;
	% RB - right bottom;
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	*/
	float area = (float)(GT.rows * GT.cols);
	x = x + 1; y = y + 1;
	LT = GT(Rect(0, 0, x, y));
	w1 = x*y / area;
	RT = GT(Rect(x, 0, GT.cols - x, y));
	w2 = (GT.cols - x)*y / area;
	LB = GT(Rect(0, y, x, GT.rows - y));
	w3 = x*(GT.rows - y) / area;
	RB = GT(Rect(x, y, GT.cols - x, GT.rows - y));
	w4 = (GT.cols - x)*(GT.rows - y) / area;

}

void DivideSM(Mat& GT, int x, int y, Mat& LT, Mat&  RT, Mat&  LB, Mat&  RB){
	/*
	% LT - left top;
	% RT - right top;
	% LB - left bottom;
	% RB - right bottom;
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	*/
	
	x = x + 1; 
	y = y + 1;

	LT = GT(Rect(0, 0, x, y));
	RT = GT(Rect(x, 0, GT.cols - x, y));
	LB = GT(Rect(0, y, x, GT.rows - y));
	RB = GT(Rect(x, y, GT.cols - x, GT.rows - y));

}


double SSIM(Mat& SM, Mat& GT){

	int N = SM.rows*SM.cols;
	
	Mat tmp_m, tmp_sd;

	meanStdDev(SM, tmp_m, tmp_sd);
	double x_mean = tmp_m.at<double>(0, 0);
	double sigma_x = tmp_sd.at<double>(0, 0);
	double sigma_x2 = pow(sigma_x, 2);

	meanStdDev(GT, tmp_m, tmp_sd);
	double y_mean = tmp_m.at<double>(0, 0);
	double sigma_y = tmp_sd.at<double>(0, 0);
	double sigma_y2 = pow(sigma_y, 2);

	//Compute the covariance between SM and GT
	Mat x_mean_mat(SM.rows, SM.cols, CV_32FC1, x_mean), y_mean_mat(GT.rows, GT.cols, CV_32FC1, y_mean);
	double sigma_xy = sum((SM - x_mean_mat).mul((GT - y_mean_mat))).val[0] / (N - 1 + eps);


	double alpha = 4 * x_mean * y_mean * sigma_xy;
	double beta = (pow(x_mean, 2) + pow(y_mean, 2)) * (sigma_x2 + sigma_y2);

	double Q = 0;
	if (alpha != 0){
		Q = alpha / (beta + eps);
	}
	else
		if (alpha == 0 && beta == 0)
			Q = 1.0;
		else
			Q = 0;

	return Q;
}


double S_region(Mat& GT, Mat& SM){
	/*
	% S_region computes the region similarity between the foreground map and
	% ground truth(as proposed in "Structure-measure:A new way to evaluate
	% foreground maps" [Deng-Ping Fan et. al - ICCV 2017])
	% Usage:
	%   Q = S_region(SM,GT)
	% Input:
	%    GT - Binary ground truth map with values in the range [0,1]. Type: double.    
	%    SM - Binary/Non-binary foreground map with values in the range [0 1]. Type: double.
	% Output:
	%   Q - The region similarity score
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	*/
	
	//find the centroid of the GT
	int x = 0, y = 0;
	centroid(GT, x, y);

	Mat GT_1, GT_2, GT_3, GT_4; 
	Mat SM_1, SM_2, SM_3, SM_4;
	float w1, w2, w3, w4;

	DivideGT(GT, x, y, GT_1, GT_2, GT_3, GT_4, w1, w2, w3, w4);
	DivideSM(SM, x, y, SM_1, SM_2, SM_3, SM_4);

	double Q1 = SSIM(SM_1, GT_1);
	double Q2 = SSIM(SM_2, GT_2);
	double Q3 = SSIM(SM_3, GT_3);
	double Q4 = SSIM(SM_4, GT_4);

	double Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4;
	return Q;
}


double Object(Mat& Mask, Mat& SM){

	double score = 0;
	if (sum(SM).val[0] <= 0)
		return score;

	Mat tmp_m, tmp_sd;
	meanStdDev(SM, tmp_m, tmp_sd, Mask);
	double x = tmp_m.at<double>(0, 0);
	double sigma_x = tmp_sd.at<double>(0, 0);

	score = 2 * x / (pow(x, 2) + 1 + sigma_x + eps);
	return score;
}


double S_object(Mat& GT, Mat& SM){
	/*
	% S_object Computes the object similarity between foreground maps and ground
	% truth(as proposed in "Structure-measure:A new way to evaluate foreground
	% maps" [Deng-Ping Fan et. al - ICCV 2017])
	% Usage:
	%   Q = S_object(GT,SM)
	% Input:
	%    GT - Binary ground truth map with values in the range [0,1]. Type: double.    
	%    SM - Binary/Non-binary foreground map with values in the range [0 1]. Type: double.
	%   
	% Output:
	%   Q - The object similarity score
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	*/

	// threshold the GT to get mask
	Mat mask;
	threshold(GT, mask, 0.5, 1, THRESH_BINARY);
	
	//The foreground and background of the GT {GT_fg, GT_bg}   
	Mat GT_fg, GT_bg;
	mask.convertTo(GT_fg, CV_8UC1);

	Mat base_bg = getStaticMat()(Rect(0, 0, SM.cols, SM.rows));
	Mat SM_bg = base_bg - SM;
	Mat not_mask = (base_bg - mask);
	not_mask.convertTo(GT_bg, CV_8UC1);

	double mu = mean(GT).val[0];

	//evaluate the similarity between GT and SM in Object-level 
    double O_fg = Object(GT_fg, SM);
	double O_bg = Object(GT_bg, SM_bg);
	double Q = mu * O_fg + (1 - mu) * O_bg;

	return Q;
}


double StructureMeasure(Mat& GT, Mat& SM){
	//set GT and SM to rang(0,1)
	normalize(GT, GT, 0, 1, NORM_MINMAX);
	normalize(SM, SM, 0, 1, NORM_MINMAX);

	//get the mean value of the GT and SM.
	double GT_mean = mean(GT).val[0];
	double SM_mean = mean(SM).val[0];
	double Q = 0;

	if (GT_mean == 0) //% if the GT is completely black
	{
		Q = 1 - SM_mean;
	}
	else{
		if (GT_mean == 1.0){//% if the GT is completely wihte
			Q = SM_mean;
		}
		else{
			double alpha = 0.5;
			double Q_object = S_object(GT, SM);
			double Q_region = S_region(GT, SM);
			Q = alpha * Q_object + (1 - alpha) * Q_region;
		}
	}
	return Q;
}

#endif