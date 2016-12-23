close all;
clear all;
clc;
%%
%%load the training image. ID 1-15 person, select 8 pictures for
%%training,declare some variates(Yale face dataset)
train_nums = 120;
test_nums = 45;
im_size = 100;
pixel_nums = 100*100;
train_data = zeros(train_nums, pixel_nums);
test_data = zeros(test_nums, pixel_nums);
train_dir =dir('E:\研究生课件\计算机视觉\project2\train_image');
test_dir = dir('E:\研究生课件\计算机视觉\project2\test_image');
train_label = [];
iteration_nums = 40;
ld = 30;

%transfer each training image to 1*10000, as a row of train_data 
for i = 1:train_nums
    image_name = sprintf('%s\\%s','E:\研究生课件\计算机视觉\project2\train_image',train_dir(i+2).name);
    im_train = imread(image_name);
    im_train = im_train(1:pixel_nums);
    train_data(i,:) = im_train;
    train_name = train_dir(i+2).name;
    train_label(i) = ceil((str2num(cell2mat(regexp(train_name,'\d','match'))))/11);  
end

for i = 1:test_nums
    image_name = sprintf('%s\\%s','E:\研究生课件\计算机视觉\project2\test_image',test_dir(i+2).name);
    im_test = imread(image_name);
    im_test = im_test(1:pixel_nums);
    test_data(i,:) = im_test; 
end

%%
%%2DLDA
k = 15;
classMean = []; 
dimension = im_size;
Srb = zeros(dimension,dimension);
Srw = zeros(dimension,dimension);
Slb = zeros(dimension,dimension);
Slw = zeros(dimension,dimension);
L = zeros(dimension,ld);
R = eye(dimension,ld);

for i = 1:k
    index = find(train_label==i);
    classMean(i,:) = mean(train_data(index, :));
end
sampleMean = mean(classMean);


for itera = 1:iteration_nums
    for i = 1:k
        index = find(train_label==i);
%compute Srw
        Xclass=train_data(index,:);
        tempSrw=zeros(dimension,dimension);
        for j=1:length(index)
            tempSrw=tempSrw + (reshape(Xclass(j,:),im_size,im_size) - reshape(classMean(i,:),im_size,im_size))*R*R'*(reshape(Xclass(j,:),im_size,im_size) - reshape(classMean(i,:),im_size,im_size))';        
        end
        Srw = Srw + tempSrw;    
    end
%compute Srb
    for i = 1:k
        Srb = Srb + length(index)*(reshape(classMean(i,:),im_size,im_size)-reshape(sampleMean,im_size,im_size))*R*R'*(reshape(classMean(i,:),im_size,im_size)-reshape(sampleMean,im_size,im_size))';
    end 

    v = inv(Srw) * Srb;
    [evec,eval] = eig(v);
    [x,d] = cdf2rdf(evec,eval);
    L = x(:,1:ld);   
    
    
    for i = 1:k
        index = find(train_label==i);
%compute Slw
        Xclass = train_data(index,:);
        tempSlw = zeros(dimension,dimension);
        for j=1:length(index)
            tempSlw = tempSlw + (reshape(Xclass(j,:),im_size,im_size)-reshape(classMean(i,:),im_size,im_size))'*L*L'*(reshape(Xclass(j,:),im_size,im_size)-reshape(classMean(i,:),im_size,im_size));        
        end
        Slw = Slw + tempSlw;    
    end
%compute Slb
    for i = 1:k
        Slb = Slb + length(index)*(reshape(classMean(i,:),im_size,im_size)-reshape(sampleMean,im_size,im_size))'*L*L'*(reshape(classMean(i,:),im_size,im_size)-reshape(sampleMean,im_size,im_size));
    end
    
    v = inv(Slw)*Slb;
    [evec,eval] = eig(v);
    [x,d] = cdf2rdf(evec,eval);
    R = x(:,1:ld);
     
end

for i = 1:k
    class_center = L'*reshape(classMean(i,:),im_size,im_size)*R;
    class_center = class_center(1:ld*ld);
    centers(i,:) = class_center;
end

%%
%%compute the accuarcy
accuracy = 0;
for i = 1:test_nums
    test_image = L'*reshape(test_data(i,:),im_size,im_size)*R;
    min = norm(test_image(1:ld*ld)-centers(1,:));
    position = 1;
    for j = 2:15
        distance = norm(test_image(1:ld*ld)-centers(j,:));
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
fprintf('Accuracy is %f,iteration_nums is %f,dimension is %f\n',accuracy,iteration_nums,ld);