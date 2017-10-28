clear;
clc;
path = 'Test/Set14/HR';
out_path  = 'Test/Set14/HR_YUV420_RGB';
image_list = dir([path, '/*.png']); 
for i = 1:14
    image = imread(fullfile(path, image_list(i).name));
    s = size(size(image));
    if(s(2) == 3)
        R = image(:, :, 1);
        G = image(:, :, 2);
        B = image(: , :, 3);
    else
        R = image;
        G = image;
        B = image;
    end
    
    [Y, U, V] = rgb2yuv(R, G, B, 'YUV420_8');
    rgb = yuv2rgb(Y, U, V, 'YUV420_8');
    imwrite(rgb, fullfile(out_path, image_list(i).name));
end