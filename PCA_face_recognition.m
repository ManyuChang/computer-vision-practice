close all;
clear all;
clc;
%%
%%load the training image. ID 1-15 person, select 8 pictures for
%%training,declare some variates
train_nums = 120;
test_nums = 45;
pixel_nums = 100*100;
Energy = 0.9;
train_data = zeros(train_nums, pixel_nums);
test_data = zeros(test_nums, pixel_nums);
train_dir =dir('E:\研究生课件\计算机视觉\project2\train_image');
test_dir = dir('E:\研究生课件\计算机视觉\project2\test_image');

%transfer each training image to 1*10000, as a row of train_data 
for i = 1:120
    image_name = sprintf('%s\\%s','E:\研究生课件\计算机视觉\project2\train_image',train_dir(i+2).name);
    im_train = imread(image_name);
    im_train = im_train(1:pixel_nums);
    train_data(i,:) = im_train; 
end

%%
%%PCA,coeff is eigenvector,latent is eigienvalue
dimension = 0;

[coeff,~,latent] = princomp(train_data);
cum_percent = cumsum(latent)/sum(latent);
for i=1:length(cum_percent)
    if cum_percent(i) >= Energy
        dimension = i;
        break;
    end
end

train_data_reduced=train_data*coeff(:,1:dimension);

%%
%%load the test images and test
for i = 1:45
    image_name = sprintf('%s\\%s','E:\研究生课件\计算机视觉\project2\test_image',test_dir(i+2).name);
    im_test = imread(image_name);
    im_test = im_test(1:pixel_nums);
    test_data(i,:) = im_test; 
end

test_data_reduced=test_data*coeff(:,1:dimension);
%%
%%compute the accuracy
accuracy = 0;
for i = 1:test_nums
    min = norm(test_data_reduced(i,:)-train_data_reduced(1,:));
    position = 1;
    for j = 2:train_nums
        distance = norm(test_data_reduced(i,:)-train_data_reduced(j,:));
        if min > distance
            min = distance;
            position = j;
        end
    end
    train_name = train_dir(position+2).name;
    test_name = test_dir(i+2).name;
    if (abs(str2num(cell2mat(regexp(test_name,'\d','match'))) - ceil(str2num(cell2mat(regexp(train_name,'\d','match')))/11)*11) <= 2)
        accuracy = accuracy + 1;
    else
        fprintf('test:%s,false match:%s\n',test_name,train_name);
    end
end

accuracy = accuracy / test_nums;
fprintf('Accuracy is %f,Energy %f,Dimension %d\n',accuracy,Energy,dimension);