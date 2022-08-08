clear all; clc; close all;
tic
source_img=double(imread('flower_160_80.png')); figure;imshow(uint8(source_img));
target_img=double(imread('flower2_160_80.png'));figure;imshow(uint8(target_img));
target_img= ~(im2gray(target_img));
[row_ind,col_ind] = find(~target_img);
[left_most_row_index,~] = min(row_ind);
[right_most_row_index,~] = max(row_ind);
[bottom_most_col_index,~] = max(col_ind);
[top_most_col_index,~] = min(col_ind);
patch_cell_r = mat2cell(source_img(:,:,1), 8*ones(1,ceil(size(source_img,1)/8)), 8*ones(1,ceil(size(source_img,2)/8)));
patch_cell_g = mat2cell(source_img(:,:,2), 8*ones(1,ceil(size(source_img,1)/8)), 8*ones(1,ceil(size(source_img,2)/8)));
patch_cell_b = mat2cell(source_img(:,:,3), 8*ones(1,ceil(size(source_img,1)/8)), 8*ones(1,ceil(size(source_img,2)/8)));
%%%%%% target image %%%%%%
patch_cell_target_r = mat2cell(target_img(:,:,1), 8*ones(1,size(target_img,1)/8), 8*ones(1,size(target_img,2)/8));
Num_NonZero_pixels = zeros(8,10);
display(size(patch_cell_target_r));
temp = [];
temp1 = [];
temp2 = [];
for i =1:8
    for j = 1:10
        x = reshape(patch_cell_target_r{i,j},64,1);
        Num_NonZero_pixels(i,j) = nnz(x);
        if Num_NonZero_pixels(i,j) == 64
            temp = [temp;Num_NonZero_pixels(i,j)];
            temp1 = [temp1;i];
            temp2 = [temp2;j];
        end
    end
end
%%%%%% Dictionary by sampling images %%%%%%
Dict_size = 79;%fog77;cat80;flower79
rng('shuffle');
r = randi([1 size(temp1,1)],1,Dict_size);
Dict_R_value = zeros(64, Dict_size);
Dict_G_value = zeros(64, Dict_size);
Dict_B_value = zeros(64, Dict_size);
for i=1:Dict_size
    p = temp1(i);
    q = temp2(i);
    x = reshape(patch_cell_r{p,q},64,1);
    Dict_R_value(:,i) = x;
    x = reshape(patch_cell_g{p,q},64,1);
    Dict_G_value(:,i) = x;
    x = reshape(patch_cell_b{p,q},64,1);
    Dict_B_value(:,i) = x;
end
%%%%%% start reconstruction index by index %%%%%%
p = 3;
q = 4;
lambda1=100; %sparsity parameter
for i = left_most_row_index:right_most_row_index
    for j = top_most_col_index:bottom_most_col_index
        data_vect_R = reshape(source_img(i-p:i+q,j-p:j+q,1),64,1);
        data_vect_G = reshape(source_img(i-p:i+q,j-p:j+q,2),64,1);
        data_vect_B = reshape(source_img(i-p:i+q,j-p:j+q,3),64,1);
        d = Dict_size;
        cvx_begin
            variable y(d)
            minimize( 0.5*norm(Dict_R_value*y-data_vect_R)+lambda1*norm(y,2))
        cvx_end
        alpha_R = y;
        cvx_begin
            variable y(d)
            minimize( 0.5*norm(Dict_G_value*y-data_vect_G)+lambda1*norm(y,2))
        cvx_end
        alpha_G = y;
        cvx_begin
            variable y(d)
            minimize( 0.5*norm(Dict_B_value*y-data_vect_B)+lambda1*norm(y,2))
        cvx_end
        alpha_B = y;

        recontructed_patch_R = reshape(Dict_R_value * alpha_R, 8, 8);
        recontructed_patch_G = reshape(Dict_G_value * alpha_G, 8, 8);
        recontructed_patch_B = reshape(Dict_B_value * alpha_B, 8, 8);
        window_patch = target_img(i-p:i+q,j-p:j+q);

        source_img(i-p:i+q,j-p:j+q,1) = source_img(i-p:i+q,j-p:j+q,1).*window_patch + recontructed_patch_R.*(~window_patch);
        source_img(i-p:i+q,j-p:j+q,2) = source_img(i-p:i+q,j-p:j+q,2).*window_patch + recontructed_patch_G.*(~window_patch);
        source_img(i-p:i+q,j-p:j+q,3) = source_img(i-p:i+q,j-p:j+q,3).*window_patch + recontructed_patch_B.*(~window_patch);
    end
end
figure;imshow(uint8(source_img));
toc


