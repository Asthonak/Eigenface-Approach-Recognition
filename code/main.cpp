#include <iostream>
/*
#include "ReadImage.cpp"
#include "WriteImage.cpp"
#include "ReadImageHeader.cpp"
#include "image.h"
#include "image.cpp"
*/
#include "Functions.cpp"

#include <vector>
#include <Eigen/Dense>

#include <dirent.h> // for reading in image directories

using namespace Eigen;
using namespace std;


int main()
{
	// variables for training/test faces, eigenfaces, eigenvalues, and the average face
	vector<pair<string, VectorXf>> trainingFaces, testFaces;
	MatrixXf eigenfaces;
	VectorXf eigenvalues;
	VectorXf avgFace;

	/***Part A***/

	// read in training/test faces
	readFaces((char*)"./fa_H", trainingFaces);
	readFaces((char*)"./fb_H", testFaces);

	// get eigenfaces, if they don't exist compute them
	if(readSavedFaces(avgFace, eigenfaces, eigenvalues, "fa_H") == false)
		computeEigenFaces(trainingFaces, avgFace, eigenfaces, eigenvalues, "fa_H");

	// normalize the eigenfaces
	normalizeEigenFaces(eigenfaces);

	// create the image of the average face
	writeFace(avgFace, (char*)"averageFace.pgm");

	// run the classfiler where PCA threshold = 80%, 90%, and 95%
	runClassifier("N-Results/NData", avgFace, eigenfaces, eigenvalues, trainingFaces, testFaces, 0.80);
	runClassifier("N-Results/NData", avgFace, eigenfaces, eigenvalues, trainingFaces, testFaces, 0.90);
	runClassifier("N-Results/NData", avgFace, eigenfaces, eigenvalues, trainingFaces, testFaces, 0.95);

	// output the eigenfaces corresponding to the 10 largest and smallest eigenvalues
	char faceFileName[100];
	for (int i = 0; i < 10; i++)
	{
		sprintf(faceFileName, "Part-AlargestFace%i.pgm", i+1);
		writeFace(eigenfaces.col(i), faceFileName);
	}

	for(int i = eigenfaces.cols() -1; i > eigenfaces.cols() -11; i--)
	{
		sprintf(faceFileName, "Part-AsmallestFace%i.pgm", (int)(i - eigenfaces.cols() + 3));
		writeFace(eigenfaces.col(i), faceFileName);
	}

	/***Part B***/

	// clear training/test faces
	trainingFaces.clear();
	testFaces.clear();

	// get eigenfaces, if they don't exist compute them
	readFaces((char*)"./fa2_H", trainingFaces);
	readFaces((char*)"./fb_H", testFaces);

	// get eigenfaces, if they don't exist compute them
	if(readSavedFaces(avgFace, eigenfaces, eigenvalues, "fa2_H") == false)
		computeEigenFaces(trainingFaces, avgFace, eigenfaces, eigenvalues, "fa2_H");

	// normalize the eigenfaces
	normalizeEigenFaces(eigenfaces);

	// create the image of the average face
	writeFace(avgFace, (char*)"averageFace-PartB.pgm");

	// run the classifier threshold function
	classifierThreshold("C-Results/CData", avgFace, eigenfaces, eigenvalues, trainingFaces, testFaces, 0.95);

	/***Part C***/

	// clear training/test faces
	trainingFaces.clear();
	testFaces.clear();

	// read in training/test faces
	readFaces((char*)"./fa_L", trainingFaces);
	readFaces((char*)"./fb_L", testFaces);

	// get eigenfaces, if they don't exist compute them
	if(readSavedFaces(avgFace, eigenfaces, eigenvalues, "fa_L") == false)
		computeEigenFaces(trainingFaces, avgFace, eigenfaces, eigenvalues, "fa_L");

	// normalize the eigenfaces
	normalizeEigenFaces(eigenfaces);

	// create the image of the average face
	writeFace(avgFace, (char*)"averageFace-PartC.pgm");

	// run the classfiler where PCA threshold = 80%, 90%, and 95%
	runClassifier("PartC-Results/CData", avgFace, eigenfaces, eigenvalues, trainingFaces, testFaces, 0.80);
	runClassifier("PartC-Results/CData", avgFace, eigenfaces, eigenvalues, trainingFaces, testFaces, 0.90);
	runClassifier("PartC-Results/CData", avgFace, eigenfaces, eigenvalues, trainingFaces, testFaces, 0.95);

	// output the eigenfaces corresponding to the 10 largest and smallest eigenvalues
	char faceFileName_c[100];
	for (int i = 0; i < 10; i++)
	{
		sprintf(faceFileName_c, "PartC-largestFace%i.pgm", i+1);
		writeFace(eigenfaces.col(i), faceFileName_c);
	}

	for(int i = eigenfaces.cols() -1; i > eigenfaces.cols() -11; i--)
	{
		sprintf(faceFileName_c, "PartC-smallestFace%i.pgm", (int)(i - eigenfaces.cols() + 3));
		writeFace(eigenfaces.col(i), faceFileName_c);
	}

	/***Part D***/

	// clear training/test faces
	trainingFaces.clear();
	testFaces.clear();

	// read in training/test faces
	readFaces((char*)"./fa2_L", trainingFaces);
	readFaces((char*)"./fb_L", testFaces);

	// get eigenfaces, if they don't exist compute them
	if(readSavedFaces(avgFace, eigenfaces, eigenvalues, "fa2_L") == false)
		computeEigenFaces(trainingFaces, avgFace, eigenfaces, eigenvalues, "fa2_L");

	// normalize the eigenfaces
	normalizeEigenFaces(eigenfaces);

	// create the image of the average face
	writeFace(avgFace, (char*)"averageFace-PartD.pgm");

	// run the classifier threshold function
	classifierThreshold("PartD-CResults/CData", avgFace, eigenfaces, eigenvalues, trainingFaces, testFaces, 0.95);

}

