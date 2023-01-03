%clear;
%close;

function [recon] = recon_fbp(gt_path, num_angle, SOD, dsensor, filter1, filter2, up_sample)

[sinogram, ~, ang] = sinogram_generate(gt_path, num_angle, SOD, dsensor, up_sample);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     TODO: Implement 2D FBP here               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[M, N]=size(sinogram);
% width of frequency domain
width = 2^nextpow2(M);  

%% FT
proj_fft = fft(sinogram, width);

% R-L filtering
if filter1
    filter = 2*[0:(width/2-1), width/2:-1:1]'/width;
else
    filter = ones(width, 1);
end

proj_filtered = zeros(width,N);
for i = 1:N
    proj_filtered(:,i) = proj_fft(:,i).*filter;
end

%figure
%subplot(1,2,1),imshow(proj_fft),title('FT of sinogram in frequency domain')
%subplot(1,2,2),imshow(proj_filtered),title('Filtered FT of sinogram in frequency domain')

%% IFT (Filtered)
proj_ifft = real(ifft(proj_filtered)); 
%figure,imshow(proj_ifft),title('Filtered sinogram in spatial domain')

% fbp with filtered sinogram
fbp = zeros(M); 
for i = 1:N
    rad = ang(i)*pi/180.;
    for x = 1:M
        for y = 1:M
            t_temp = (x-M/2) * cos(rad) - (y-M/2) * sin(rad)+M/2  ;
             % nearest interpolation
            t = round(t_temp) ;
            if t>0 && t<=M
                fbp(x,y)=fbp(x,y)+proj_ifft(t,i);
            end
        end
    end
end
recon=fbp;
% laplace filter
if filter2
    kernel = [0, 1, 0; 1, -4, 1; 0, 1, 0];
    recon = imfilter(recon, kernel, 'replicate');
end
% normalize
recon = (recon - min(min(recon))) / (max(max(recon)) - min(min(recon)));
figure
fig = imshow(recon);
title('reconstructed image');
saveas(fig, ['./outputdata/recon_', num2str(num_angle), '.png']);
close();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                 END OF YOUR CODE                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%niftiwrite(recon, recon_path);