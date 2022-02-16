%Ethan Che
%Homework 3
%11/04/2021
%I pledge my honor that I have abided by the Stevens Honor System.

function main (filename1, filename2, stdv)

    %First image
    
    img1 = imread(filename1);

    gray1 = rgb2gray(img1);
    
    rowsize1 = size(img1, 1);
    colsize1 = size(img1, 2);

    %First, gauss filter the image

    stdv = str2num(stdv);
    
    temp = 2*floor(stdv/2)+1; %Standard deviation, rounded to next odd number
    width = 6*temp;
    width = 2*floor(width/2)+1;
    
    add = floor(width/2); %How many pixels have to add aropund edge. Change the first number.

    extended1 = extend_edges(gray1,add);
    gauss_filtered1 = gauss_filter(rowsize1, colsize1, extended1,stdv, width);

    gauss_e1 = extend_edges(gauss_filtered1, 1);

    [dx1, dy1] = sobel_filter(rowsize1, colsize1, gauss_e1);
    dx_e1 = extend_edges(dx1, 1);
    dy_e1 = extend_edges(dy1, 1);

    [response1,features1, top1k1] = harris(dx_e1, dy_e1, rowsize1, colsize1, 1);
    imwrite(response1, 'harris_left.png');
    imwrite(top1k1, 'top1k_left.png');

    %Second image
    
    img2 = imread(filename2);
    gray2 = rgb2gray(img2);
    rowsize2 = size(img2, 1);
    colsize2 = size(img2, 2);

    %First, gauss filter the image

    extended2 = extend_edges(gray2,add);
    gauss_filtered2 = gauss_filter(rowsize2, colsize2, extended2,stdv, width);

    gauss_e2 = extend_edges(gauss_filtered2, 1);

    [dx2, dy2] = sobel_filter(rowsize2, colsize2, gauss_e2);
    dx_e2 = extend_edges(dx2, 1);
    dy_e2 = extend_edges(dy2, 1);

    [response2,features2, top1k2] = harris(dx_e2, dy_e2, rowsize2, colsize2, 1);
    imwrite(response2, 'harris_right.png');
    imwrite(top1k2, 'top1k_right.png');

    cast1  = cast(response1,'uint8');
    cast2 = cast(response2,'uint8');

    [ssd, l1, l2] = apply_ssd(gray1, gray2, cast1, cast2, 19, rowsize1, colsize1, rowsize2, colsize2, features1, features2);
    subplot(2,2,1);
    showMatchedFeatures(img1,img2,l1,l2,'montage');
    title('Matched Features');
    A_matrix = affine_1(l1,l2, 3, 15);

    pano1 = display(A_matrix, img1, img2);
    subplot(2,2,3);
    imshow(cast(pano1,'uint8'));
    title('A1');

    A_matrix2 = affine_2(ssd,l1,l2,3, 15); 
    pano2 = display(A_matrix2, img1, img2);
    subplot(2,2,4);
    imshow(cast(pano2,'uint8'));
    title('A1+A2');
    
    

end

function pano = display(A_matrix, color1, color2)
    tform = maketform('affine', A_matrix);
    [J, xdata, ydata] = imtransform(color1,tform);

    new_x = round(xdata(2)-xdata(1));
    new_y = round(ydata(2)-ydata(1));

    shift_x = -1*xdata(1); %How much we have to shift the right image over by
    shift_y = -1*ydata(1);
    shift_x = round(shift_x);
    shift_y = round(shift_y);
    pano = zeros(new_y, new_x);
    pano(:,:,1) = 0;
    pano(:,:,2) = 0;
    pano(:,:,3) = 0;
    
    %Add the right image in the correct spot
    for i = 1:size(color2,1)
        for j = 1:size(color2,2)
            
            loc_x = j+shift_x;
            loc_y = i+shift_y;
            
            pano(loc_y,loc_x,:) = color2(i,j,:);
        end
    end
    
    %Add in the left image
    for i = 1:size(J,1)
        for j = 1:size(J,2)
            if (J(i,j,:) ~= [0 0 0])
                if (pano(i,j,:) == [0 0 0])
                    pano(i,j,:) = J(i,j,:);
                else
                    pano(i,j,:) = double(J(i,j,:).*(0.5)) + double(pano(i,j,:).*(0.5));
                end
            end
        end
    end
end

function [suppressed, num_features, top1k] = harris(dx, dy, im_r, im_c, offset)
    
    response = zeros(im_r, im_c); %Holds detected corners
    %Compute second moment matrix around each pixel in 3x3 area.
    for i = 1+offset:im_r+offset
        for j = 1+offset:im_c+offset
            
            xx_m = dx((i-offset):(i+offset), (j-offset):(j+offset)).^2; %top left in second moment matrix
            yy_m = dy((i-offset):(i+offset), (j-offset):(j+offset)).^2; %bottom right in second moment matrix
            xy_m = dx((i-offset):(i+offset), (j-offset):(j+offset)) * dy((i-offset):(i+offset), (j-offset):(j+offset)); %TR/L
            m = [xx_m xy_m; xy_m yy_m];
            temp = det(m)-.05*trace(m)^2;
            if (temp < 1e15)
                response(i-offset, j-offset) = 0;
            else
                response(i-offset, j-offset) = temp;
            end

        end
    end

    %Now, only keep the 1000 highest values
    is = [];
    js = [];
    vals = [];
    for n = 1:1000
        [i j] = find(response==max(max(response)));
        is = [is i(1)];
        js = [js j(1)];
        vals = [vals response(i(1),j(1))];
        response(i(1),j(1)) = 0;
    end

    top1k = zeros(im_r, im_c);

    for n = 1:1000
        i = is(n);
        j = js(n);
        val = vals(n);
        top1k(i,j) = val;
    end

    %Apply NMS
    
    suppressed = nms_filter(top1k);

    num_features = 0;

    for i = 1:im_r
        for j = 1:im_c
            if (suppressed(i,j) ~= 0)
                num_features = num_features + 1;
            end
        end
    end
end

function final_matrix = affine_1(l1,l2, distance_thresh, points_thresh) %affine using A1
    %l1 and l2 contain the x,y of top 20 features from SSD
    
    %Pick three random correspondences. Estimate the transform. Then for
    %the other features, see how well that transform holds up. Distance
    %from actual point to calculated position of point. If small enough,
    %count it as an inlier.

    num_inliers = 0;
    num_iterations = 0;
    a = 0;
    b = 0;
    c = 0;
    d = 0;
    e = 0;
    f = 0;

    avg_err = 0;

    while (num_inliers < points_thresh) %While you have to keep looking
        num_inliers = 0;
        num_iterations = num_iterations+1;
        avg_err = 0;
        %Pick three random points
        ranints = randperm(20,3);
        x1 = l1(ranints(1),1);
        y1 = l1(ranints(1),2);
        x1p = l2(ranints(1),1);
        y1p = l2(ranints(1),2);

        x2 = l1(ranints(2),1);
        y2 = l1(ranints(2),2);
        x2p = l2(ranints(2),1);
        y2p = l2(ranints(2),2);

        x3 = l1(ranints(3),1);
        y3 = l1(ranints(3),2);
        x3p = l2(ranints(3),1);
        y3p = l2(ranints(3),2);

        

        matrix_A = [x1 y1 1 0 0 0; 0 0 0 x1 y1 1; x2 y2 1 0 0 0; 0 0 0 x2 y2 1; x3 y3 1 0 0 0; 0 0 0 x3 y3 1];
        
        
        matrix_b = [x1p; y1p; x2p; y2p; x3p; y3p];

        

        matrix_x = linsolve(matrix_A, matrix_b); %Hypothesized transform

        
        a = matrix_x(1);
        b = matrix_x(2);
        c = matrix_x(3);
        d = matrix_x(4);
        e = matrix_x(5);
        f = matrix_x(6);

        %Now, check how good of a transform it is.
        for i = 1:20
            x = l1(i,1); %x and y of left image
            y = l1(i,2);
            temp_matrix = [a b c; d e f];
            xys = [x; y; 1];
            primes = temp_matrix*xys; %3x1 matrix that holds xprime and yprime

            xp = primes(1,1); %calcualted x and y of right image
            yp = primes(2,1);

            x_act = l2(i,1); %actual x and y of right image
            y_act = l2(i,2);

            dist = abs(norm( [x_act;y_act] - [xp;yp]));
            avg_err = avg_err+dist;
            if (dist<distance_thresh)
                num_inliers = num_inliers + 1;
            end

        end
        avg_err = avg_err/20;

    end

    final_matrix = [a d 0; b e 0; c f 1];
    disp(matrix_x);
    disp("Average error for a1: " + avg_err);
    disp("Number of iterations for a1: " + num_iterations);
end

function final_matrix = affine_2(ssd, l1, l2, distance_thresh, points_thresh) %affine using A1 and A2

    %l1 and l2 contain the x,y of top 20 features from SSD
    %ssd contains all features, of which we will select 30 at random

    %Find 30 random correspondences
    ranints = randperm(size(ssd,1),30);
    ranpoints = [];
    for i = 1:30
        ranpoints = [ranpoints; ssd(ranints(i),:)];
    end
    %disp(ranpoints);

    t1 = [];
    t2 = [];

    for i = 1:30
        t1 = [t1; [ranpoints(i,1:2)] ];
        t2 = [t2; [ranpoints(i,3:4)] ];
    end
    
    %Add the random points to l1 and l2
    l1 = [l1; t1];
    l2 = [l2; t2];

    num_inliers = 0;
    num_iterations = 0;
    a = 0;
    b = 0;
    c = 0;
    d = 0;
    e = 0;
    f = 0;

    avg_err = 0;

    while (num_inliers < points_thresh) %While you have to keep looking
        num_iterations = num_iterations + 1;
        num_inliers = 0;
        avg_err = 0;
        %Pick three random points
        ranints = randperm(20,3);
        x1 = l1(ranints(1),1);
        y1 = l1(ranints(1),2);
        x1p = l2(ranints(1),1);
        y1p = l2(ranints(1),2);

        x2 = l1(ranints(2),1);
        y2 = l1(ranints(2),2);
        x2p = l2(ranints(2),1);
        y2p = l2(ranints(2),2);

        x3 = l1(ranints(3),1);
        y3 = l1(ranints(3),2);
        x3p = l2(ranints(3),1);
        y3p = l2(ranints(3),2);

        

        matrix_A = [x1 y1 1 0 0 0; 0 0 0 x1 y1 1; x2 y2 1 0 0 0; 0 0 0 x2 y2 1; x3 y3 1 0 0 0; 0 0 0 x3 y3 1];
        
        
        matrix_b = [x1p; y1p; x2p; y2p; x3p; y3p];

        

        matrix_x = linsolve(matrix_A, matrix_b); %Hypothesized transform

        
        a = matrix_x(1);
        b = matrix_x(2);
        c = matrix_x(3);
        d = matrix_x(4);
        e = matrix_x(5);
        f = matrix_x(6);

        %Now, check how good of a transform it is.
        for i = 1:50
            x = l1(i,1); %x and y of left image
            y = l1(i,2);
            temp_matrix = [a b c; d e f];
            xys = [x; y; 1];
            primes = temp_matrix*xys; %3x1 matrix that holds xprime and yprime

            xp = primes(1,1); %calcualted x and y of right image
            yp = primes(2,1);

            x_act = l2(i,1); %actual x and y of right image
            y_act = l2(i,2);

            dist = abs(norm( [x_act;y_act] - [xp;yp]));
            avg_err = avg_err+dist;
            if (dist<distance_thresh)
                num_inliers = num_inliers + 1;
            end

        end
        avg_err = avg_err/50;

    end


    final_matrix = [a d 0; b e 0; c f 1];
    disp(matrix_x);
    disp("Average error for a1+a2: " + avg_err);
    disp("Number of iterations for a1+a2: " + num_iterations);
end

function [ssd, xy1_l, xy2_l] = apply_ssd(gray1, gray2, im1, im2, patch_size, im_r1, im_c1, im_r2, im_c2, f1, f2) %Patch size should be odd, floor(n/2). Number of additional spaces next to middle
    %For example, patch size of 1 would imply a 3x3 area.
    extended1 = extend_edges(im1, patch_size);
    extended2 = extend_edges(im2, patch_size);
    gray1 = extend_edges(gray1, patch_size);
    gray2 = extend_edges(gray2, patch_size);
    
    ssd = zeros(f1*f2, 5); %Each row corresponds to a feature match. Each
    %row has 5 elements, the x's and y's of the two feature locations and
    %the value of the ssd
    n = 1;
    %Row/col of ssd correspond to where a feature was detected in im1. At
    %each position, value will be a list in such order: [a b val], where
    %these values are defined below.
    %Look for a feature in im1
    for i = 1+patch_size:im_r1+patch_size
        for j = 1+patch_size:im_c1+patch_size
            %Find a feature in im1
            if (extended1(i,j) == 255) %if (extended1(i,j) ~= 0)
                patch1 = gray1((i-patch_size):(i+patch_size), (j-patch_size):(j+patch_size));
                %Now, compare against every feature in im2
                for a = 1+patch_size:im_r2+patch_size
                    for b = 1+patch_size:im_c2+patch_size
                        %Find a feature in im2
                        if (extended2(a,b) == 255) %if (extended2(a,b) ~= 0)
                            patch2 = gray2((a-patch_size):(a+patch_size), (b-patch_size):(b+patch_size));
                            %Now, compute the SSD
                            diff = patch1-patch2;
                            val = sum(diff(:).^2);
                            ssd(n,1) = i-patch_size;
                            ssd(n,2) = j-patch_size;
                            ssd(n,3) = a-patch_size;
                            ssd(n,4) = b-patch_size;
                            ssd(n,5) = val;
                            n=n+1;
                        end
                    end
                end
            end
        end
    end
    
    %Now, find the top 20 (smallest ssd's)

    working_copy = ssd(:,:);
    xy1_l = [];
    xy2_l = [];
    usedx = [];
    usedy = [];
    max_elem = max(working_copy(:,5));

    for n = 1:20
        min = max_elem;
        x1 = 0;
        y1 = 0;
        x2 = 0;
        y2 = 0;
        for i = 1:size(working_copy,1)
            if (working_copy(i,5) < min && working_copy(i,5) ~= -1)
                min = working_copy(i,5);
                x1 = working_copy(i,2);
                y1 = working_copy(i,1);
                x2 = working_copy(i,4);
                y2 = working_copy(i,3);
                working_copy(i,5) = -1;
            end
        end
        xy1_l = [xy1_l ; [x1 y1] ];
        xy2_l = [xy2_l ; [x2 y2] ];
    end
end

%Helpers-------------------------------------------------------------

function filtered_image = gauss_filter(im_r, im_c, extended_image, stdv, fsz) %fsz is the filter size, 5 means 5x5, for example
    filtered_image = zeros(im_r, im_c); %Size of regular image
    mask = zeros(fsz, fsz);
    mid = ceil(fsz/2); %Center of matrix. fsv should be odd
    total = 0;
    for i = 1:fsz
        for j = 1:fsz
            temp = gauss_eq(stdv, i-mid, j-mid);
            mask(i,j) = temp;
            total = total+temp;
        end
    end
    mask = mask./total;
    %disp(sum(sum(mask)));
    offset = floor(fsz/2); %Half of the filter width rounded down
    for i = 1:im_r
        for j = 1:im_c
            %move filter over each pixel in extended image, where original
            %image is
            filtered_image(i, j) = get_filter_val(extended_image,i+offset,j+offset,mask,offset);
        end
    end
end

function gauss = gauss_eq(stdv, x,y) %Calculates the value of the gaussian distribution, with x and y as offsets from center
    a = 1/(2*pi*(stdv*stdv));
    pow = (x*x)+(y*y);
    pow = pow/(-2*stdv*stdv);
    gauss = a*exp(pow);
end

function val = get_filter_val(image, r, c, filter, offset)
    val = 0;
    subimg = image((r-offset):(r+offset), (c-offset):(c+offset)); %Part of extended image that is being filtered
    val = sum(dot(double(subimg),double(filter)));
end

function extended = extend_edges(img, add) %Extends image so we can apply filters


    
    rowsize = size(img, 1);
    colsize = size(img, 2);
    top = zeros(add,colsize); %Top pixels
    bottom = zeros(add,colsize); %Bottom pixels
    left = zeros(rowsize, add); %Left pixels
    right = zeros(rowsize, add); %Right pixels
    
    %Expand top and bottom row of pixels
    for c = 1:colsize
        for r = 1:add
            top(r,c) = img(1,c);
            bottom(r,c) = img(rowsize,c);
        end
    end
    
    for c = 1:add
        for r = 1:rowsize
            left(r,c) = img(r,1);
            right(r,c) = img(r,colsize);
        end
    end
    
    %Now, need to add the four corners where the pixels weren't extended
    tl = top(1,1); %Top left corner value
    tr = top(1,colsize); %Top right corner value
    bl = bottom(1,1); %Bottom left corner value
    br = bottom(1,colsize); %Bottom right corner value
    
    %Arrays that hold the corner pixels
    atl = zeros(add,add);
    atr = zeros(add,add);
    abl = zeros(add,add);
    abr = zeros(add,add);
    for i = 1:add
        for j = 1:add
            atl(i,j) = tl;
            atr(i,j) = tr;
            abl(i,j) = bl;
            abr(i,j) = br;
        end
    end
    extended = ones(rowsize+add+add, colsize+add+add); %Image extended out
    extended = extended.*-1; %Initialize each element to -1 so we know which elements have been changed
    
    top = [atl top atr];
    bottom = [abl bottom abr];
    %disp(bottom);
    %Add top row
    for i = 1:add
        for j = 1:size(extended,2)
            extended(i,j) = top(i,j);
        end
    end
    %Add bottom row
    for i = (size(extended,1)-add+1):size(extended,1)
        r = 1;
        for j = 1:size(extended,2)
            extended(i,j) = bottom(r,j);
        end
        r = r+1;
    end
    %Add left and right rows
    for i = 1+add:size(extended,1)-add
        for j = 1:add
            extended(i,j) = left(i-add,j);
        end
        c = 1;
        for j = (size(extended,2)-add+1):size(extended,2)
            extended(i,j) = right(i-add,c);
        end
        c = c+1;
    end
    %Add back in original image
    for i = 1:rowsize
        for j = 1:colsize
            extended(i+add,j+add) = img(i,j);
        end
    end
    
end

function [dx, dy] = sobel_filter(im_r, im_c, extended_image) %Gets sobel val of pixel
    %Also returns the matrix of gradient directions for each pixel, to be
    %used in NMS
    dx = zeros(im_r, im_c);
    dy = zeros(im_r, im_c);
    
    mask_x = [-1 0 1; -2 0 2; -1 0 1];
    mask_y = [1 2 1; 0 0 0; -1 -2 -1];
    offset = 1; %Sobel filter is 3x3, so offset is only 1
    for i = 1:im_r
        for j = 1:im_c
            %move filter over each pixel in extended image, where original
            %image is
            xval = get_filter_val(extended_image,i+offset,j+offset,mask_x,offset);
            yval = get_filter_val(extended_image,i+offset,j+offset,mask_y,offset);
            dx(i,j) = abs(double(xval));
            dy(i,j) = abs(double(yval));
        end
    end
end

function suppressed = nms_filter(edges)

    im_r = size(edges, 1);
    im_c = size(edges, 2);
    suppressed = zeros(im_r, im_c);

    %extend edges so we can check each pixel surrounding a pixel
    add = 1;
    extended_edges = extend_edges(edges, add);
    for i = 1+add:im_r+add
        for j = 1+add:im_c+add
            curr = extended_edges(i,j);
            tl = extended_edges(i-1, j-1);
            tm = extended_edges(i-1,j);
            tr = extended_edges(i-1,j+1);
            left = extended_edges(i,j-1);
            right = extended_edges(i,j+1);
            bl = extended_edges(i+1, j-1);
            bm = extended_edges(i+1,j);
            br = extended_edges(i+1,j+1);
            
            arr = [curr tl tm tr left right bl bm br];
            m = max(arr);
            if (curr == m)
                suppressed(i-add,j-add) = curr;
            end
        end
    end
    
end
