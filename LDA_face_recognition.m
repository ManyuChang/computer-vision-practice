close all;
clear all;
clc;
%%
%%load the training image. ID 1-15 person, select 8 pictures for
%%training,declare some variates(Yale face dataset)
train_nums = 120;
test_nums = 45;
pixel_nums = 100*100;
Energy = 0.9;
train_data = zeros(train_nums, pixel_nums);
test_data = zeros(test_nums, pixel_nums);
train_dir =dir('E:\研究生课件\计算机视觉\project2\train_image');
test_dir = dir('E:\研究生课件\计算机视觉\project2\test_image');
train_label = [];

%transfer each training image to 1*10000, as a row of train_data 
for i = 1:120
    image_name = sprintf('%s\\%s','E:\研究生课件\计算机视觉\project2\train_image',train_dir(i+2).name);
    im_train = imread(image_name);
    im_train = im_train(1:pixel_nums);
    train_data(i,:) = im_train;
    train_name = train_dir(i+2).name;
    train_label(i) = ceil((str2num(cell2mat(regexp(train_name,'\d','match'))))/11);  
end

for i = 1:45
    image_name = sprintf('%s\\%s','E:\研究生课件\计算机视觉\project2\test_image',test_dir(i+2).name);
    im_test = imread(image_name);
    im_test = im_test(1:pixel_nums);
    test_data(i,:) = im_test; 
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
test_data_reduced=test_data*coeff(:,1:dimension);


%%
%%LDA
k = 15;
classMean = []; 
Sb = zeros(dimension,dimension);
Sw = zeros(dimension,dimension);
for i = 1:k
    index = find(train_label==i);
    classMean(i,:) = mean(train_data_reduced(index, :));
%compute Sw
    Xclass=train_data_reduced(index,:);
    tempSw=zeros(dimension,dimension);
    for j=1:length(index)
        tempSw=tempSw+(Xclass(j,:)-classMean(i,:))'*(Xclass(j,:)-classMean(i,:));        
    end
    Sw=Sw + tempSw;    
end
sampleMean = mean(classMean);
%compute Sb
 for i = 1:k
    Sb = Sb + length(index)*(classMean(i,:)-sampleMean)'*(classMean(i,:)-sampleMean);
 end

v = inv(Sw) * Sb;
[evec,eval] = eig(v);
[x,d] = cdf2rdf(evec,eval);
W = x(:,1:k-1); 

for i = 1:k
    index = find(train_label==i);
    Xclass=train_data_reduced(index,:);    
    centers(i,:)=mean(Xclass*W);
end
 
train_lda = train_data_reduced * W;
test_lda = test_data_reduced * W;

%%
%%compute the accuarcy

accuracy = 0;
for i = 1:test_nums
    min = norm(test_lda(i,:)-centers(1,:));
    position = 1;
    for j = 2:15
        distance = norm(test_lda(i,:)-centers(j,:));
        if min > distance
            min = distance;
            position = j;
        end
    end
    test_name = test_dir(i+2).name;
    if (abs(str2num(cell2mat(regexp(test_name,'\d','match'))) - ceil(position*11)) <= 2)
        accuracy = accuracy + 1;
    else
        fprintf('test:%s,false label:%d\n',test_name,position);
    end
end

accuracy = accuracy / test_nums;
fprintf('Accuracy is %f\n',accuracy);