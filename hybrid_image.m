clear all
close all


%% load image
im1 = imread('data/motorcycle.bmp');
im2 = imread('data/bicycle.bmp');
figure,imshow(im1);title('im1');
figure,imshow(im2);title('im2');
[m, n, ch] = size(im1);

%% transfer images into frequency
im1_F = fftshift(fft2(im1));
im2_F = fftshift(fft2(im2));


%% design filter
d1 = 20;
d2 = 30;
M = fix(m/2);
N = fix(n/2);
for c = 1:3
for i = 1:m
    for j = 1:n
        if sqrt((i-M)^2+(j-N)^2) >= d1
            h1 = 0;
        else
            h1 = 1;
        end
             im1_F(i,j,c) = h1 * im1_F(i,j,c);
       
        if sqrt((i-M)^2+(j-N)^2) >= d2
            h2 = 0;
        else
            h2 = 1;
        end
             im2_F(i,j,c) = h2 * im2_F(i,j,c);
    end
end
end
%%
im1_result = uint8(real(ifft2(ifftshift(im1_F))));
figure,imshow(im1_result);title('im1');
im2_result = im2 - uint8(real(ifft2(ifftshift(im2_F))));
figure,imshow(im2_result);title('im2');
im_result = im1_result + im2_result;
figure,imshow(im_result);title('im');
            
