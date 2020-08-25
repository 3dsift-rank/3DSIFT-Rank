# 3DSIFT-Rank

## Publication

<a href="https://doi.org/10.1016/j.neuroimage.2019.116208">Chauvin, L., Kumar, K., Wachinger, C., Vangel, M., de Guise, J., Desrosiers, C., Wells, W., Toews, M. and Alzheimer’s Disease Neuroimaging Initiative, 2020. Neuroimage signature from salient keypoints is highly specific to individuals and shared by close relatives. NeuroImage, 204, p.116208.</a>

## FeatExtract
featExtract is the program used to extract SIFT-Ranked [1] features from images. It accepts nifti images (.nii, .hdr, .nii.gz) as input, and output a list of keypoints and their descriptors.

#### Usage
Volumetric local feature extraction v1.1  
Usage: **featExtract [options] \<input image\> \<output features\>**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\<input image\>: nifti (.nii,.hdr,.nii.gz).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\<output features\>: output file with features.  

#### Options 
 <table>
  <tr>
    <td>-w</td><td>Output feature geometry in world coordinates, NIFTI qto_xyz matrix (default is voxel units).</td> 
  </tr>
  <tr>
    <td>-2+</td><td>Double input image size.</td>
  </tr>
  <tr>
    <td>-2-</td><td>Halve input image size.</td>
  </tr>
 </table>
 
 #### Output
 The program will output a .key file (with the same name as the input file), containing a list of features, with their coordinates, scale, orientation, and descriptor.
 
## FeatMatchMultiple
featMatchMultiple is the program, based on FLANN library [2], used to match features.

#### Usage
Volumetric Feature matching v1.1  
Usage: **featMatchMultiple [options] -f \<input filelist\>**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\<input filelist\>: Text file containing the list of .key files (from featExtract). 1 key file per line.
    
#### Option
 <table>
  <tr>
    <td>-n</td><td>Number of nearest neighbors. Default: 5.</td> 
  </tr>
   <tr>
    <td>-r-</td><td>Use only unrotated features.</td> 
  </tr>
 </table>

#### Output
The program will output different files:

 <table>
  <tr>
    <td>_command.txt</td><td>The complete command used to generate this results (for logging purpose).</td> 
  </tr>
  <tr>
    <td>_names.txt</td><td>List of the input image filename</td> 
  </tr>
  <tr>
    <td><b>matching_votes.txt</b></td><td>NxN matrix containing the accumulation of the weighted votes for each pair of image. <br><b>The most important file.</b></td> 
  </tr>
  <tr>
    <td>feature_count.txt</td><td>The number of features for each input file.</td> 
  </tr>
  <tr>
    <td>vote_count.txt</td><td>NxN matrix containing the accumulation of the non-weighted votes for each pair of image.</td> 
  </tr>
 </table>

# Data
https://central.xnat.org/data/projects/SIFTFeatures

# References
[1] Toews, Matthew, and William Wells. "Sift-rank: Ordinal description for invariant feature correspondence." 2009 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2009.  
[2] Muja, Marius, and David G. Lowe. "Scalable nearest neighbor algorithms for high dimensional data." IEEE transactions on pattern analysis and machine intelligence 36.11 (2014): 2227-2240.

# Contact
laurent.chauvin0@gmail.com, matt.toews@gmail.com 
