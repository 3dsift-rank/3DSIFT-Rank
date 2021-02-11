/////////////////////////////////////////////////////
//                                                 //
// File: featMatchUtilities.cpp                    //
// Author: Matthew Toews                           //
// Contributor: Laurent Chauvin                    //
//                                                 //
/////////////////////////////////////////////////////

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <assert.h>
#include <functional>
#include "MultiScale.h"
#include <time.h>
#include <algorithm>
#include <vector>
#include <map>
#include <cmath>

#define INCLUDE_FLANN
#ifdef INCLUDE_FLANN
	#include "flann/flann.h"
#endif

using namespace std;

int
removeNonReorientedFeatures(
	vector<Feature3DInfo> &vecFeats
)
{
	int iRemoveCount = 0;
	for (int i = 0; i < vecFeats.size(); i++)
	{
		if (!(vecFeats[i].m_uiInfo & INFO_FLAG_REORIENT))
		{
			vecFeats.erase(vecFeats.begin() + i);
			i--;
			iRemoveCount++;
		}
	}
	return iRemoveCount;
}

int
removeReorientedFeatures(
	vector<Feature3DInfo> &vecFeats
)
{
	int iRemoveCount = 0;
	for (int i = 0; i < vecFeats.size(); i++)
	{
		if ((vecFeats[i].m_uiInfo & INFO_FLAG_REORIENT))
		{
			vecFeats.erase(vecFeats.begin() + i);
			i--;
			iRemoveCount++;
		}
		else
		{
			memset(&vecFeats[i].ori[0][0], 0, sizeof(vecFeats[i].ori));
			vecFeats[i].ori[0][0] = 1;
			vecFeats[i].ori[1][1] = 1;
			vecFeats[i].ori[2][2] = 1;
		}
	}
	return iRemoveCount;
}

#ifdef INCLUDE_FLANN

typedef struct _msSearchStructure
{
	// FLANN parameters
	struct FLANNParameters g_flann_params;
	float g_speedup;
	flann_index_t g_index_id;
	int g_nn;

	// Size of vector 
	int iVectorSize;

	// Data vectors
	float *g_pf1; // New image feature
	float *g_pf2; // Data base of features
	int	*g_piFeatureFrequencies; // Frequency count array, one for each feature
	int g_iFeatCount; // Count of all features
	int *g_piFeatureLabels; // Label array, one for each feature, currently set to image index
        int *g_piFeatureImageIndex; // Image index array, one for each feature 
  
							// Label vectors
	vector< float > vfLabelCounts; // A prior over all discrete label values, e.g. in training data

								   // Coordinates / Scale for each feature
	float *g_pfX;
	float *g_pfY;
	float *g_pfZ;
	float *g_pfS;

	int	*g_piFeatureL0; // quick label counter
	int	*g_piFeatureL1; // quick label counter

						// Index result
	int *g_piResult; // Result array for one search (input features x g_nn neighours)
	float *g_pfDist;  // Distance array for one search (input features x g_nn neighours)
	int g_iMaxImageFeatures; // Max size of array for one search

							 // Indices of features g_pf1
	int *g_piImgIndices; // Index array, one for each image, index mapping image to feature arrays
	int *g_piImgFeatureCounts; // Count array, one for each image, number of features per image
	int g_iImgCount;

	// file output
	FILE *outfile;

	// Labels
} msSearchStructure;

msSearchStructure g_SS;

//
// msInitNearestNeighborApproximate()
//
// Init for NN search for self
//
int
msNearestNeighborApproximateInit(
	vector< vector<Feature3DInfo> > &vvFeats,
	int iNeighbors,
	vector< int > &vLabels,
	float fGeometryWeight,
	int iImgSplit
)
{
	g_SS.iVectorSize = PC_ARRAY_SIZE;
	if (fGeometryWeight > 0)
	{
		// Add coordinates to vectors: add 
		g_SS.iVectorSize = PC_ARRAY_SIZE + 3;
	}

	// Tells where to split train / test data
	int iFeatureSplit = 0;

	printf("Descriptor size: %d\n", g_SS.iVectorSize);

	g_SS.g_flann_params = DEFAULT_FLANN_PARAMETERS;
	g_SS.g_flann_params.algorithm = FLANN_INDEX_KDTREE;
	g_SS.g_flann_params.trees = 8;
	g_SS.g_flann_params.log_level = FLANN_LOG_INFO;
	g_SS.g_flann_params.checks = 64;
	g_SS.g_flann_params.sorted = 1;
	g_SS.g_flann_params.cores = 1;
	g_SS.g_nn = iNeighbors;

	// Per-image info/arrays: feature indices and counts
	g_SS.g_iImgCount = vvFeats.size();
	g_SS.g_piImgIndices = new int[vvFeats.size()];
	g_SS.g_piImgFeatureCounts = new int[vvFeats.size()];
	g_SS.g_iFeatCount = 0;
	g_SS.g_iMaxImageFeatures = 0;
	for (int i = 0; i < vvFeats.size(); i++)
	{
		if (i == iImgSplit)
			iFeatureSplit = g_SS.g_iFeatCount;

		g_SS.g_piImgIndices[i] = g_SS.g_iFeatCount;
		g_SS.g_piImgFeatureCounts[i] = vvFeats[i].size();
		g_SS.g_iFeatCount += vvFeats[i].size();
		if (vvFeats[i].size() > g_SS.g_iMaxImageFeatures)
		{
			g_SS.g_iMaxImageFeatures = vvFeats[i].size();
		}

		while (vLabels[i] >= g_SS.vfLabelCounts.size())
		{
			// push back zero bins
			g_SS.vfLabelCounts.push_back(0);
		}
		// Add number of features associated with this label
		g_SS.vfLabelCounts[vLabels[i]] += vvFeats[i].size();
	}

	float fPriorSum = 0;
	for (int i = 0; i < g_SS.vfLabelCounts.size(); i++)
	{
		// Add an extra sample to normalize distribution
		g_SS.vfLabelCounts[i]++;
		fPriorSum += g_SS.vfLabelCounts[i];
	}
	// Normalize label distribution - divide by total labels (images) in addition to prior (number of labels)
	for (int i = 0; i < g_SS.vfLabelCounts.size(); i++)
	{
		g_SS.vfLabelCounts[i] /= fPriorSum;
	}

	// Per-feature arrays: labels and descriptors
	g_SS.g_iFeatCount = g_SS.g_iFeatCount;
	g_SS.g_piFeatureLabels = new int[g_SS.g_iFeatCount];
	g_SS.g_piFeatureImageIndex = new int[g_SS.g_iFeatCount];
	g_SS.g_pf2 = new float[g_SS.g_iFeatCount*g_SS.iVectorSize];  // Array to hold entire feature set
	g_SS.g_piFeatureFrequencies = new int[g_SS.g_iFeatCount];  // Array to hold frequency counts for each input feature
	g_SS.g_piFeatureL0 = new int[g_SS.g_iFeatCount]; // Quick label counters
	g_SS.g_piFeatureL1 = new int[g_SS.g_iFeatCount];
	memset(g_SS.g_piFeatureL0, 0, sizeof(int)*g_SS.g_iFeatCount);
	memset(g_SS.g_piFeatureL1, 0, sizeof(int)*g_SS.g_iFeatCount);

	g_SS.g_pfX = new float[g_SS.g_iFeatCount];
	g_SS.g_pfY = new float[g_SS.g_iFeatCount];
	g_SS.g_pfZ = new float[g_SS.g_iFeatCount];
	g_SS.g_pfS = new float[g_SS.g_iFeatCount];

	// Per-result arrays: indices and distances
	g_SS.g_piResult = new int[g_SS.g_iMaxImageFeatures*g_SS.g_nn];
	g_SS.g_pfDist = new float[g_SS.g_iMaxImageFeatures*g_SS.g_nn];
	g_SS.g_pf1 = NULL; // Array to hold max features from one image


	// Copy descriptors into array
	int iFeatCount = 0;
	for (int i = 0; i < vvFeats.size(); i++)
	{
		for (int j = 0; j < vvFeats[i].size(); j++)
		{
			memcpy(g_SS.g_pf2 + iFeatCount*g_SS.iVectorSize, &(vvFeats[i][j].m_pfPC[0]), sizeof(vvFeats[i][j].m_pfPC));

			// This is where the feature info goes - set to zero for debugging
			if (fGeometryWeight > 0)
			{
				g_SS.g_pf2[iFeatCount*g_SS.iVectorSize + 0] = fGeometryWeight*vvFeats[i][j].x;
				g_SS.g_pf2[iFeatCount*g_SS.iVectorSize + 1] = fGeometryWeight*vvFeats[i][j].y;
				g_SS.g_pf2[iFeatCount*g_SS.iVectorSize + 2] = fGeometryWeight*vvFeats[i][j].z;

				g_SS.g_pf2[iFeatCount*g_SS.iVectorSize + 0] /= vvFeats[i][j].scale;
				g_SS.g_pf2[iFeatCount*g_SS.iVectorSize + 1] /= vvFeats[i][j].scale;
				g_SS.g_pf2[iFeatCount*g_SS.iVectorSize + 2] /= vvFeats[i][j].scale;
			}

			//** add geometry weights here: small shift

			g_SS.g_piFeatureLabels[iFeatCount] = vLabels[i];
			g_SS.g_piFeatureImageIndex[iFeatCount] = i;
			g_SS.g_piFeatureFrequencies[iFeatCount] = 0; // Set to zero

			g_SS.g_pfX[iFeatCount] = vvFeats[i][j].x;
			g_SS.g_pfY[iFeatCount] = vvFeats[i][j].y;
			g_SS.g_pfZ[iFeatCount] = vvFeats[i][j].z;
			g_SS.g_pfS[iFeatCount] = vvFeats[i][j].scale;

			iFeatCount++;
		}
	}

	if (iImgSplit <= 0 || iFeatureSplit <= 0)
		iFeatureSplit = g_SS.g_iFeatCount;

	g_SS.g_index_id = flann_build_index(g_SS.g_pf2, iFeatureSplit, g_SS.iVectorSize, &g_SS.g_speedup, &g_SS.g_flann_params);

	g_SS.outfile = fopen("report.all.txt", "wt");
	return 1;
}

int
msNearestNeighborApproximateDelete(
)
{
	if (g_SS.outfile) fclose(g_SS.outfile);
	flann_free_index(g_SS.g_index_id, &g_SS.g_flann_params);
	delete[] g_SS.g_piFeatureFrequencies;
	delete[] g_SS.g_pfDist;
	delete[] g_SS.g_piResult;
	delete[] g_SS.g_pf2;

	return 1;
}


//
// msNearestNeighborApproximateSearchSelf()
//
// Search a specific image. This function is used in class
//
int
msNearestNeighborApproximateSearchSelf(
	int iImgIndex,
	vector< int > &vfImgCounts,  // Size: #images in database. Return value: # of times features match to database image N
	vector< float > &vfLabelCounts, // Size: #labels in database. Label counts large enough to fit all labels
	vector< float > &vfLabelLogLikelihood, // Size: #labels in database Log likelihood, large enough to fit all labels
	float** ppfMatchingVotes,
	int** ppiLabelVotes
)
{
	// Subtract this image label from label prior distribution (add it later)
	// This is important particularly labels with few samples, e.g. nearest neighbors
	int iImgLabel = g_SS.g_piFeatureLabels[g_SS.g_piImgIndices[iImgIndex]];
	float fImgFeaturesProb = g_SS.g_piImgFeatureCounts[iImgIndex] / ((float)(g_SS.g_iFeatCount + g_SS.vfLabelCounts.size()));
	g_SS.vfLabelCounts[iImgLabel] -= fImgFeaturesProb;

	float *pfPriorProbsCT = &(g_SS.vfLabelCounts[0]);

	for (int j = 0; j < vfImgCounts.size(); j++)
		vfImgCounts[j] = 0;
	for (int j = 0; j < vfLabelLogLikelihood.size(); j++)
		vfLabelLogLikelihood[j] = 0;

	int* piResult = new int[g_SS.g_iMaxImageFeatures*g_SS.g_nn];
	float* pfDist = new float[g_SS.g_iMaxImageFeatures*g_SS.g_nn];
	float* pf1 = g_SS.g_pf2 + g_SS.g_piImgIndices[iImgIndex] * g_SS.iVectorSize;
	
	flann_find_nearest_neighbors_index(g_SS.g_index_id, pf1, g_SS.g_piImgFeatureCounts[iImgIndex], piResult, pfDist, g_SS.g_nn, &g_SS.g_flann_params);

	int iOutOfBounds = 0;

	// No deviation on location filter - if location is relevant (aligned subjects, e.g. brain),
	// then use concatenated geometry
	float fMaxDeviationLocation = 10000000;
	std::map<int, float> votedFeatures;
	for (int i = 0; i < g_SS.g_piImgFeatureCounts[iImgIndex]; i++)
	{
		// Initialize counts with Laplacian prior - actually Laplacian is no good, need actual prior
		for (int j = 0; j < vfLabelCounts.size(); j++)
			vfLabelCounts[j] = 1.0*pfPriorProbsCT[j];

		int iQueryFeatIndex = g_SS.g_piImgIndices[iImgIndex] + i;

		// Identify min distance to neighbor, use this as stdev in Gaussian weighting scheme
		float fMinDist = -1;
		std::vector< std::pair<int, float> > indexNN;
		std::vector< int > matchingImages;
		for (int j = 0; j < g_SS.g_nn; j++)
		{
			int iResultFeatIndex = piResult[i*g_SS.g_nn + j];
			int iLabel = g_SS.g_piFeatureLabels[iResultFeatIndex];
			int iImage = g_SS.g_piFeatureImageIndex[iResultFeatIndex];

			// Geometrical consistency check - not used, fMaxDeviationLocation set to infinity
			if (
				fabs(g_SS.g_pfX[iQueryFeatIndex] - g_SS.g_pfX[iResultFeatIndex]) < fMaxDeviationLocation &&
				fabs(g_SS.g_pfY[iQueryFeatIndex] - g_SS.g_pfY[iResultFeatIndex]) < fMaxDeviationLocation &&
				fabs(g_SS.g_pfZ[iQueryFeatIndex] - g_SS.g_pfZ[iResultFeatIndex]) < fMaxDeviationLocation &&
				1
				)
			{
				// Result index must not be from the query image
				if (iResultFeatIndex < g_SS.g_piImgIndices[iImgIndex] || iResultFeatIndex > g_SS.g_piImgIndices[iImgIndex] + g_SS.g_piImgFeatureCounts[iImgIndex])
				{
					// Only count features from other images
					if (indexNN.size() < g_SS.g_nn)
					{
						// Ensure feature only vote once per image
						if (std::find(matchingImages.begin(), matchingImages.end(), iImage) == matchingImages.end())
						{
							/* A vote does not already exists for this image */
							std::pair<int, float> tmp(piResult[i*g_SS.g_nn + j], pfDist[i*g_SS.g_nn + j]);
							indexNN.push_back(tmp);

							if (fMinDist == -1 || pfDist[i*g_SS.g_nn + j] < fMinDist)
							{
								if (pfDist[i*g_SS.g_nn + j] > 0)
								{
									// For duplicated scans, distance will be 0, we want first non-null minimum distance
									fMinDist = pfDist[i*g_SS.g_nn + j];
								}
							}

							matchingImages.push_back(iImage);
						}
					}
					else {
						break;
					}
				}
			}
			else
			{
				// Should not get here with huge permissible deviation
				assert(0);
			}
		}


		// Normalize weights
		float fSumWeights = 0;
		std::vector<float> weights;
		for (int j = 0; j < indexNN.size(); ++j)
		{
			int iResultFeatIndex = indexNN[j].first;

			float fDx = (g_SS.g_pfX[iQueryFeatIndex] - g_SS.g_pfX[iResultFeatIndex]);
			float fDy = (g_SS.g_pfY[iQueryFeatIndex] - g_SS.g_pfY[iResultFeatIndex]);
			float fDz = (g_SS.g_pfZ[iQueryFeatIndex] - g_SS.g_pfZ[iResultFeatIndex]);
			float fSc1 = g_SS.g_pfS[iQueryFeatIndex];
			float fSc2 = g_SS.g_pfS[iResultFeatIndex];

			// ----------------------------------------------------------------------------------------
			// Appearance Weight

			float fDistApp = indexNN[j].second;
			float fDistSqApp = fDistApp*fDistApp;

			float fVarApp = fMinDist*fMinDist;

			float fAppWeight = std::exp(-fDistSqApp / fVarApp);

			// ----------------------------------------------------------------------------------------
			// Geometrical Weight

			float fScn = 25;

			// 'Gravitational' Distance
			float fDistSqGeom = fDx*fDx + fDy*fDy + fDz*fDz;

			float fVarGeom = fSc1*fSc2;

			float fGeoWeight = std::exp(-fDistSqGeom / (fVarGeom*fVarGeom));

			// ----------------------------------------------------------------------------------------
			// Scale Weight

			float fDistSqScale = std::pow(std::log(fSc1) - std::log(fSc2), 2);
			
			float fVarScale = 1;
			
			float fScaleWeight = std::exp(-fDistSqScale / fVarScale);

			// ----------------------------------------------------------------------------------------
			// Total Weight
			
			float eta = 1; // Background distribution

			float fTotalWeight = fAppWeight * fGeoWeight * fScaleWeight;
			fTotalWeight += eta;
			fTotalWeight = std::log(fTotalWeight);
			fTotalWeight /= std::log(eta + 1);

			weights.push_back(fTotalWeight);
		}
		
		// Now add results based on min distance neighbor 
		//for( int j = 0; j < g_SS.g_nn; j++ )
		for (int j = 0; j < indexNN.size(); ++j)
		{
			//int iResultFeatIndex = g_SS.g_piResult[i*g_SS.g_nn+j];
			int iResultFeatIndex = indexNN[j].first;
			int iLabel = g_SS.g_piFeatureLabels[iResultFeatIndex];
			int iImage = g_SS.g_piFeatureImageIndex[iQueryFeatIndex];

			float fDx = (g_SS.g_pfX[iQueryFeatIndex] - g_SS.g_pfX[iResultFeatIndex]);
			float fDy = (g_SS.g_pfY[iQueryFeatIndex] - g_SS.g_pfY[iResultFeatIndex]);
			float fDz = (g_SS.g_pfZ[iQueryFeatIndex] - g_SS.g_pfZ[iResultFeatIndex]);
			float fDistGeom = fDx*fDx + fDy*fDy + fDz*fDz;
			if (fDistGeom > 0)
				fDistGeom = sqrt(fDistGeom);


			// Geometrical consistency check
			if (
				fabs(g_SS.g_pfX[iQueryFeatIndex] - g_SS.g_pfX[iResultFeatIndex]) < fMaxDeviationLocation &&
				fabs(g_SS.g_pfY[iQueryFeatIndex] - g_SS.g_pfY[iResultFeatIndex]) < fMaxDeviationLocation &&
				fabs(g_SS.g_pfZ[iQueryFeatIndex] - g_SS.g_pfZ[iResultFeatIndex]) < fMaxDeviationLocation &&
				1
				)
			{
				if (iResultFeatIndex < g_SS.g_piImgIndices[iImgIndex] || iResultFeatIndex > g_SS.g_piImgIndices[iImgIndex] + g_SS.g_piImgFeatureCounts[iImgIndex])
				{
					// Only count features from other images
					float fDist = pfDist[i*g_SS.g_nn + j];
					float fExponent = fDist / (fMinDist + 1.0);
					// Gaussian weighting
					float fValue = exp(-fExponent*fExponent) / pfPriorProbsCT[iLabel];
					vfLabelCounts[iLabel] += fValue;

					// Multiple features from same subject cannot match to same feature in another subject (e.g. A1 -> B1, A2 -> B1)
					if (votedFeatures.find(iResultFeatIndex) != votedFeatures.end())
					{
						float previousVote = votedFeatures[iResultFeatIndex];

						// If this vote is better, delete previous one
						if (weights[j] > previousVote)
						{

							if (previousVote > 0) {
								ppfMatchingVotes[iImage][iLabel] -= previousVote;
							}
							ppfMatchingVotes[iImage][iLabel] += weights[j];
							votedFeatures[iResultFeatIndex] = weights[j];
						}
					}
					else
					{
						// Store votes
						ppfMatchingVotes[iImage][iLabel] += weights[j];
						ppiLabelVotes[iImage][iLabel] += 1;
						votedFeatures.insert({ iResultFeatIndex, weights[j] });
					}

					vfImgCounts[iLabel]++;
				}
			}
			else
			{
				// Should not get here with huge permissible deviation
				assert(0);
			}
		}

		// Compute & accumulate log likelihood
		float fTotal = 0;
		for (int j = 0; j < vfLabelCounts.size(); j++)
		{
			// Divide here by prior probability of a sample
			fTotal += vfLabelCounts[j];
		}
		// Note that normalization here makes no difference immediately
		for (int j = 0; j < vfLabelCounts.size(); j++)
		{
			vfLabelLogLikelihood[j] += log(vfLabelCounts[j] / (float)fTotal);
		}
	}

	// Add label back on
	g_SS.vfLabelCounts[iImgLabel] += fImgFeaturesProb;

	delete[] piResult;
	delete[] pfDist;
	
	return 1;
}

#endif
