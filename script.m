gt_path = './inputdata/gt.nii';
SOD = 380;
dsensor = 0.1;
num_angles = [45, 90, 180, 360];

% sinogram     : each column of sinogram represents the 1D projection result from the
% pos          : the 1D sensor position array
% ang          : the angles of projection
% filter1      : R-L filter to sparse-views
% filter2      : laplacian filter to reconstruct image
% up_sample    : dense-view interpolated to 360 views

filter1=false;
filter2=false;
up_sample=true;
for num_angle = num_angles
    [recon] = recon_fbp(gt_path, num_angle, SOD, dsensor, filter1, filter2, up_sample);
end 