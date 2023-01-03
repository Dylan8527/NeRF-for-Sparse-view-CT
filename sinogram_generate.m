%clear;
%close;

function [sparse_image, sensor_pos, sino] = sinogram_generate(gt_path, num_angle, SOD, dsensor, up_sample)
% configuration for fan-beam geometry
% num_angle: num of projection angles
% SOD      : source object distance
% dsensor  : spacing between sensors in the detector array.

% gt;
img = niftiread(gt_path);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            TODO: Implement ifanbeam algrithm here             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%info = niftiinfo(gt_path)
%imagesize = info.ImageSize
[F, sensor_pos, Fangles] = fanbeam(img, SOD, 'FanSensorSpacing', dsensor);
step = int32(360/num_angle);
sino = [];
for i = 0:num_angle-1
  sino = [sino,Fangles(i*step+1)+1];
end
sparse_image = [];
for i = 1:num_angle
  sparse_image = [sparse_image,F(:,sino(i))];
end

sinogram = sparse_image;
if up_sample % up sample angles to 720
    ang = sino;
    new_sinogram = zeros(size(sinogram, 1), 360);
    left_ang     = ang(1:end-1);
    right_ang    = ang(2:end);
    mid_ang      = (left_ang + right_ang) /2.;
    for an = 1:360
        diff = (mid_ang - an) .^ 2;
        [~, index] = sort(diff, 2, 'ascend');
        l_ang = ang(index(1));
        r_ang = ang(index(1)+1);
        lambda = r_ang - l_ang;
        new_sinogram(:, an) = ((r_ang - an).*sinogram(:, index(1)) + (an-l_ang).*sinogram(:, index(1)+1)) / lambda;
    end
    sparse_image = new_sinogram;
    sino = 1:360;
end 

% show views if True
figure
fig = imshow(sparse_image,[],'XData',sino,'YData',sensor_pos,...
            'InitialMagnification','fit');
axis normal
%xlabel('Rotation Angles (degrees)')
%xlim([0, max(sino)])
%ylabel('Sensor Positions')
%ylim([min(sensor_pos), max(sensor_pos)])
%axis on
%colormap(gca,hot), colorbar
saveas(fig, ['./outputdata/proj_', num2str(num_angle), '.png']);
close();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        END OF YOUR CODE                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
base_dir = './inputdata/';
if up_sample
     niftiwrite(sparse_image,[base_dir, 'dense_', num2str(num_angle), '_sino.nii']);
    niftiwrite(sensor_pos, [base_dir, 'dense_', num2str(num_angle), '_pos.nii']);
    niftiwrite(sino, [base_dir, 'dense_', num2str(num_angle), '_ang.nii']);
else
    niftiwrite(sparse_image,[base_dir, 'sparse_', num2str(num_angle), '_sino.nii']);
    niftiwrite(sensor_pos, [base_dir, 'sparse_', num2str(num_angle), '_pos.nii']);
    niftiwrite(sino, [base_dir, 'sparse_', num2str(num_angle), '_ang.nii']);
end