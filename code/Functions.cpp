#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdio.h>

//#include "jacobi.cpp"
#include "ReadImage.cpp"
#include "WriteImage.cpp"
#include "ReadImageHeader.cpp"
#include "image.h"
#include "image.cpp"

#include <vector>
#include <Eigen/Dense>

#include <dirent.h> // for reading in image directories

using namespace Eigen;
using namespace std;

// functions to read and write binary files using Eigen
namespace Eigen
{
	template<class Matrix>
	void write_binary(const char* filename, const Matrix& matrix)
	{
		std::ofstream out(filename, ios::out | ios::binary | ios::trunc);
		typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
		out.write((char*) (&rows), sizeof(typename Matrix::Index));
		out.write((char*) (&cols), sizeof(typename Matrix::Index));
		out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
		out.close();
	}

	template<class Matrix>
	void read_binary(const char* filename, Matrix& matrix)
	{
		std::ifstream in(filename, ios::in | std::ios::binary);
		typename Matrix::Index rows = 0, cols = 0;
		in.read((char*) (&rows), sizeof(typename Matrix::Index));
		in.read((char*) (&cols), sizeof(typename Matrix::Index));
		matrix.resize(rows, cols);
		in.read((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar));
		in.close();
	}
}

void readFaces(char *filepath, vector<pair<string, VectorXf>> &faces)
{
	DIR *directory;
	struct dirent *ent;

	// read in all images in the directory
	directory = opendir(filepath);
	while ((ent = readdir(directory)) != NULL)
	{
		if(ent->d_name[0] != '.')
		{
			int rows, cols, levels;
			bool type;
			char name[100] = "";

			//read in each individual image
			strcat(name, filepath);
			strcat(name, "/"); 
			strcat(name, ent->d_name); 
			readImageHeader(name, rows, cols, levels, type);
			ImageType currentImage(rows, cols, levels);

			readImage(name, currentImage);

			VectorXf currentFace = VectorXf(rows*cols);
			for(int i = 0; i < rows; i++)
			{
				for(int j = 0; j < cols; j++)
				{
					int temp = 0;
					currentImage.getPixelVal(i, j, temp);
					currentFace[i*cols + j] = temp;
				}
			}
			// store in the faces vector
			faces.push_back(pair<string, VectorXf>(string(ent->d_name, 5), currentFace));
		}
	}

	closedir(directory);
}

void writeFace(VectorXf face, char *fileName)
{
	// determine the number of rows/cols/levels of the image
	int rows, cols;
	if(face.rows() == 320)
	{
		rows = 20;
		cols = 16;
	}
	if(face.rows() == 2880)
	{
		rows = 60;
		cols = 48;
	}
	int levels = 255;
	ImageType img(rows, cols, levels);

	// find the min/max
	float min = face.minCoeff();
	float max = face.maxCoeff();

	// determin the values of each pixel
	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < cols; j++)
		{
			float val = (face[i*cols + j] - min) / (max - min);
			img.setPixelVal(i, j, val*255);
		}
	}

	// write the image
	writeImage(fileName, img);
}



void computeEigenFaces(vector<pair<string, VectorXf> > trainingFaces, VectorXf &averageFace, MatrixXf &eigenFaces, VectorXf &eigenValues, const char *path)
{
	char fileName[100];
	EigenSolver<MatrixXf> solver;
	MatrixXf A;
	ofstream output;

	MatrixXf eigenVectors;

	// compute the average face
	averageFace = VectorXf(trainingFaces[0].second.rows());
	averageFace.fill(0);

	for(auto it = trainingFaces.begin(); it != trainingFaces.end(); it++)
	{
		averageFace += (*it).second;
	}

	averageFace /= trainingFaces.size();

	// store to output file
	sprintf(fileName, "%s-avg-binary.dat", path);

	Eigen::write_binary(fileName, averageFace);

	A = MatrixXf(averageFace.rows(), trainingFaces.size());
	for(vector<VectorXf>::size_type i = 0; i < trainingFaces.size(); i++)
	{
		A.col(i) = trainingFaces[i].second - averageFace;
	}

	eigenVectors = MatrixXf(trainingFaces.size(), trainingFaces.size());
	eigenVectors = A.transpose()*A;
	solver.compute(eigenVectors, true);

	// compute the eigenfaces
	eigenFaces = MatrixXf(averageFace.rows(), trainingFaces.size());
	eigenFaces = A * solver.eigenvectors().real();

	// compute the eigenvalues
	eigenValues = VectorXf(eigenFaces.cols());
	eigenValues = solver.eigenvalues().real();

	// store to output files
	sprintf(fileName, "%s-binary.dat", path);
	Eigen::write_binary(fileName, eigenFaces);
	Eigen::read_binary(fileName, eigenFaces);

	sprintf(fileName, "%s-values-binary.dat", path);
	Eigen::write_binary(fileName, eigenValues); 

	sprintf(fileName, "%s-EigenVectors.txt", path);
	output.open(fileName);
	output << eigenFaces;
	output.close();
}

// function to normalize eigenfaces
void normalizeEigenFaces(MatrixXf &eigenfaces)
{
	for(int i = 0; i < eigenfaces.cols(); i++)
	{
		eigenfaces.col(i).normalize();  
	}
}

bool readSavedFaces(VectorXf &averageFace, MatrixXf &eigenfaces, VectorXf &eigenvalues, const char *filepath)
{
	char filename[100];

	ifstream inputStream;

	// check if eigenfaces binary file exists
	sprintf(filename, "%s-binary.dat", filepath);
	inputStream.open(filename);
	if(inputStream.fail())
	{
		return false;
	}
	else
	{
		Eigen::read_binary(filename, eigenfaces);
	}
	inputStream.close();

	// check if eigenvalues binary file exists
	sprintf(filename, "%s-values-binary.dat", filepath);
	inputStream.open(filename);
	if(inputStream.fail())
	{
		return false;
	}
	else
	{
		Eigen::read_binary(filename, eigenvalues);
	}
	inputStream.close();

	// check if eigenvalues average face file exists
	sprintf(filename, "%s-avg-binary.dat", filepath);
	inputStream.open(filename);
	if(inputStream.fail())
	{
		return false;
	}
	else
	{
		Eigen::read_binary(filename, averageFace);
	}
	inputStream.close();

	return true;
}

// compares the second element of the pair
bool compare(pair<string, float> a, pair<string, float> b)
{
    return a.second < b.second;
}

VectorXf projectOntoEigenspace(VectorXf newFace, VectorXf averageFace, MatrixXf eigenfaces)
{
	//calculte the projection of the new face on the new face
	vector<float> faceCoefficients;
	VectorXf normalizedFace = newFace - averageFace;
	VectorXf projectedFace(averageFace.rows());
	projectedFace.fill(0);
	for (int i = 0; i < eigenfaces.cols(); i++)
	{
		float a = (eigenfaces.col(i).transpose() * normalizedFace)(0, 0);
		faceCoefficients.push_back(a);
		projectedFace += (faceCoefficients[i] * eigenfaces.col(i));
	}
	return projectedFace + averageFace;
}

bool amongNMostSimilarFaces(vector<pair<string, float>> similarFaces, int N, string searchID)
{
	// check if search ID maches ID of of any top n similar faces
	for(int i=0; i<N; i++)
	{
		if(similarFaces[i].first == searchID)
		{
			return true;
		}
	}
	return false;
}

void runClassifier(const char* resultsFilepath, VectorXf averageFace, MatrixXf eigenfaces, VectorXf eigenvalues, vector<pair<string, VectorXf>> trainingFaces, vector<pair<string, VectorXf>> testFaces, float PCA_percentage)
{
	// PCA dimensionality reduction
	float eigenvalues_sum = eigenvalues.sum();
	float currentEigenTotal = 0;
	int count;
	char filename[100];

	// find number of vectors to the percentatge of info given by PCA_percentage
	for(count = 0; currentEigenTotal / eigenvalues_sum < PCA_percentage && count < eigenvalues.rows(); count++)
	{
		currentEigenTotal += eigenvalues.row(count)(0);
	}

	//reduce dimansionality from # of eigenface cols to value given by count
	MatrixXf reducedEigenfaces(averageFace.rows(), count);
	reducedEigenfaces = eigenfaces.block(0, 0, averageFace.rows(), count);

	// project faces on reduced eigenfaces
	vector<pair<string, VectorXf>> projectedTrainingFaces, projectedTestFaces;
	for(int i=0; i<trainingFaces.size(); i++)
	{
		pair<string, VectorXf> temp(trainingFaces[i].first, projectOntoEigenspace(trainingFaces[i].second, averageFace, reducedEigenfaces));
		projectedTrainingFaces.push_back(temp);
	}
	for(int i=0; i<testFaces.size(); i++)
	{
		pair<string, VectorXf> temp(testFaces[i].first, projectOntoEigenspace(testFaces[i].second, averageFace, reducedEigenfaces));
		projectedTestFaces.push_back(temp);
	}

	// find correct and incorrect classifications
	VectorXf projectedTestFace;
	int correct = 0;
	int incorrect = 0;
	bool querySaved = false;
	ofstream output;
	vector<float> N_Performances(50, 0);

	sprintf(filename, "%s-%i-NImageNames.txt", resultsFilepath, (int)(PCA_percentage*100));
	output.open(filename);

	// check each query face for if it can be classified correctly
	for(int i=0; i<testFaces.size(); i++)
	{
		projectedTestFace = projectedTestFaces[i].second;
		// <image id, distance>
		vector< pair<string, float> > queryPairs;

		// check if correct or incorrect
		querySaved = false;

		// find distances from each training face to this test face and sort them in accending order
		for(int j=0; j<trainingFaces.size(); j++)
		{
			pair<string, float> newPair(trainingFaces[j].first, (projectedTestFace - projectedTrainingFaces[j].second).norm());
			queryPairs.push_back(newPair);
		}
		sort(queryPairs.begin(), queryPairs.end(), compare);
		
		for(int n=0; n<50; n++)
		{
			if(amongNMostSimilarFaces(queryPairs, n+1, projectedTestFaces[i].first)) // make func
			{
				N_Performances[n] += 1;
				// save correct match if N = 1
				if(correct < 3 && !querySaved && n == 0)
				{
					output << "Correct Test Image " << correct << " ID: " 
					<< testFaces[i].first;
					output << " Correct Training Image " << correct << " ID: " 
					<< queryPairs[0].first;
					output << endl << endl;
					correct++;
					querySaved = true;
				}
			}
			else
			{
				// save incorrect match if N = 1
				if(incorrect < 3 && !querySaved && n == 0)
				{
					output << "Incorrect Test Image " << incorrect << " ID: " << testFaces[i].first;
					output << " Incorrect Training Image " << incorrect << " ID: " << queryPairs[0].first;
					output << endl << endl;
					incorrect++;
					querySaved = true;
				}
			}
		}
	}
	output.close();

	// write data for CMC curve to output file
	sprintf(filename, "%s-%i.txt", resultsFilepath, (int)(PCA_percentage*100));
	output.open(filename);
	for(int n = 0; n < 50; n++)
	{
		output << n+1 << "\t" << (N_Performances[n] / (float)testFaces.size()) << endl;
	}
	output.close();
}

void classifierThreshold(const char* resultsFilepath, VectorXf averageFace, MatrixXf eigenfaces, VectorXf eigenvalues, vector<pair<string, VectorXf> > trainingFaces, vector<pair<string, VectorXf> > queryFaces, float PCA_percentage)
{
	// PCA dimensionality reduction
	float eigenValuesSum = eigenvalues.sum();
	float currentEigenTotal = 0;
	int count;
	char fileName[100];
	
	// find number of vectors to the percentatge of info given by PCA_percentage
	for(count = 0; currentEigenTotal / eigenValuesSum < PCA_percentage && count < eigenvalues.rows(); count++)
	{
		currentEigenTotal += eigenvalues.row(count)(0);
	}

	cout << "Dimensionality reduced from " << eigenfaces.cols() << " to " << count << endl;

	//reduce dimansionality from # of eigenface cols to value given by count
	MatrixXf reducedEigenFaces(averageFace.rows(), count);
	reducedEigenFaces = eigenfaces.block(0,0,averageFace.rows(),count);
	
	// project faces on reduced eigenfaces
	vector<pair<string, VectorXf> > projectedTrainingFaces, projectedQueryFaces;

	for(int i = 0; i < queryFaces.size(); i++)
	{
		pair<string, VectorXf> temp(queryFaces[i].first, projectOntoEigenspace(queryFaces[i].second, averageFace, reducedEigenFaces));
		projectedQueryFaces.push_back(temp);
	}

	// find true positives and false positives
	VectorXf projQueryFace;
	int TPCount, FPCount;
	TPCount = 0;
	FPCount = 0;

	pair<int, int> temp(0, 0);
	
	vector< pair<int, int> > counts(1800, temp);

	for (int i = 0; i < projectedQueryFaces.size(); i++)
	{
		cout << "\rQuery Face: " << i;
		projQueryFace = projectedQueryFaces[i].second;
		vector<pair<string, float> > queryPairs;

		for (int t = 0; t < trainingFaces.size(); t++)
		{
			pair<string, float> newPair(trainingFaces[t].first, (projQueryFace - trainingFaces[t].second).norm());
			queryPairs.push_back(newPair);
		}
		sort(queryPairs.begin(), queryPairs.end(), compare);
		cout << "\t" << queryPairs[0].second << endl;
		
		//for(int threshold = 380; threshold < 1500; threshold += 5) // high res
		for (int threshold = 50; threshold < 600; threshold += 2) // low res
		{
			if(queryPairs[0].second <= threshold)
			{
				// true positive case
				if (atoi(projectedQueryFaces[i].first.c_str()) > 50)
				{
					counts[threshold].first++;
				}
				// false positive case
				else
				{
					counts[threshold].second++;
				}
			}
		}
	
	}

	// output results
	sprintf(fileName, "%s-%i.txt", resultsFilepath, (int)(PCA_percentage*100));
	ofstream output;

	output.open(fileName);

	//for(int threshold = 380; threshold < 1500; threshold += 5) // high res
	for(int threshold = 50; threshold < 600; threshold += 2) // low res
	{
		float TPRate = (float)counts[threshold].first / (float)trainingFaces.size();
		float FPRate = (float)counts[threshold].second / ((float)queryFaces.size() - (float)trainingFaces.size());

		output << threshold << "\t" << TPRate << "\t" << FPRate << endl;
	}

	output.close();

}
