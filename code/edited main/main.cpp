#include <iostream>

#include "jacobi.cpp"
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

void readFaces(char *filepath, vector<pair<string, VectorXf>> &faces)
{
	DIR *directory;
	struct dirent *ent;

	directory = opendir(filepath);
	while ((ent = readdir(directory)) != NULL)
	{
		if(ent->d_name[0] != '.')
		{
			int rows, cols, levels;
			bool type;
			char name[100] = "";

			strcat(name, filepath);
			strcat(name, "/"); 
			strcat(name, ent->d_name); 
			readImageHeader(name, rows, cols, levels, type);
			ImageType currentImage(rows, cols, levels);

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
			faces.push_back(pair<string, VectorXf>(string(ent->d_name, 5), currentFace));
		}
	}

	closedir(directory);
}

void computeEigenFaces(vector<pair<string, VectorXf> > trainingFaces, VectorXf &averageFace, MatrixXf &eigenFaces, VectorXf &eigenValues, const char *path)
{
	char fileName[100];
	EigenSolver<MatrixXf> solver;
	MatrixXf A;
	ofstream output;

	MatrixXf eigenVectors;

	averageFace = VectorXf(trainingFaces[0].second.rows());
	averageFace.fill(0);
	for(auto it = trainingFaces.begin(); it != trainingFaces.end(); it++)
	{
		averageFace += (*it).second;
	}

	averageFace /= trainingFaces.size();

	sprintf(fileName, "%s-avg-binary.dat", path);

	//Eigen::write_binary(fileName, averageFace);

	A = MatrixXf(averageFace.rows(), trainingFaces.size());
	for(vector<VectorXf>::size_type i = 0; i < trainingFaces.size(); i++)
	{
		A.col(i) = trainingFaces[i].second - averageFace;
	}

	eigenVectors = MatrixXf(trainingFaces.size(), trainingFaces.size());
	eigenVectors = A.transpose()*A;
	solver.compute(eigenVectors, true);

	eigenFaces = MatrixXf(averageFace.rows(), trainingFaces.size());

	eigenFaces = A * solver.eigenvectors().real();

	eigenValues = VectorXf(eigenFaces.cols());

	eigenValues = solver.eigenvalues().real();

	sprintf(fileName, "%s-values-binary.dat", path);

	//Eigen::write_binary(fileName, eigenValues); //need to define this along with read_binary

	sprintf(fileName, "%s-EigenVectors,txt", path);
	output.open(fileName);
	output << eigenFaces;
	output.close();
}

int main()
{
	vector<pair<string, VectorXf>> trainingFaces, testFaces;
	MatrixXf eigenfaces;
	VectorXf eigenvalues;
	VectorXf avgFace;

	readFaces((char*)"./fa_H", trainingFaces);
	readFaces((char*)"./fb_H", testFaces);
}

