import cv2 
import numpy as np
import stitcher2 as sti
import matplotlib.pyplot as plt
from skimage.segmentation import slic
import csv
from os import path
import os
import time

global stitch
stitch = sti.Stitcher()

def find_max_dim(src_imgs, dst_img, H_list):
    #Compute set of image corners
    K = np.shape(src_imgs)[0]
    d = np.shape(src_imgs)[1]
    #K -= 1
    #d -= 1

    min_x =0
    min_y =0
    max_x =d
    max_y  = K

    for i in range(len(H_list)):
    	K,d = np.shape(src_imgs[i])[0:2]
    	tmp_corners = np.mat([[d,0,1],[0,K,1],[d,K,1]]).T
    	tmp_corners = np.dot(H_list[i], tmp_corners)
    	tmp_corners /= tmp_corners[2,:]


    	tmp_max = np.max(tmp_corners,axis=1)
    	tmp_min = np.min(tmp_corners, axis=1)

    	min_x = min(tmp_min[1,0],min_x)
    	max_x = max(tmp_max[1,0],max_x)
    	min_y = min(tmp_min[0,0],min_y)
    	max_y = max(tmp_max[0,0],max_y)

    x_dim = max_x - min_x + 1
    y_dim = max_y - min_y + 1
    coord_shift = [int(-min_y), int(-min_x)]

    return (int(y_dim), int(x_dim), 3), coord_shift
	

def combineSegments(segments):
	cc = 1
	out = np.zeros(segments[0].shape)
	for i in range(len(segments)):
		vals = np.unique(segments[i])
		for j in vals:
			if j != 0:
				ids = (segments[i] == j)
				out[ids] = cc
				cc = cc + 1

	return out

def performBlending2(images):
	
	# Add images together (as floats)
	summed_img = np.zeros(images[0].shape).astype('float')
	pixel_count = np.zeros(images[0].shape)
	for i in range(len(images)):

		summed_img = summed_img + images[i]
		# Count Number of nonzero
		pixel_count = pixel_count + (images[i] > 0).astype('float')

	# set zeros to 1.
	pixel_count = (pixel_count == 0) + pixel_count


	# divide sum by nonzero
	out = np.divide(summed_img, pixel_count)

	return out.astype('uint8')

def performBlending(images):
	print("Performing Blending...")
	m,n,_ = np.shape(images[0])
	blended = np.zeros((m,n,3))
	for i in range(m):
		for j in range(n):
			num_nonzero = 0
			out_pixel = [0,0,0]
			for k in range(len(images)):
				if np.any(images[k][i,j,:] > 0):
					num_nonzero += 1
					out_pixel[0] += images[k][i,j,0]
					out_pixel[1] += images[k][i,j,1]
					out_pixel[2] += images[k][i,j,2]
			if num_nonzero > 0:
   			    blended[i,j,:] = np.array(out_pixel)/num_nonzero

	return blended.astype('uint8')



def blend_images(src_img, dst_img, segmentsL, H_list):
    # compute max pano size. 
    #pano_shape, dst_origin = find_max_dim(src_img, dst_img, H_list)
    #final_pano = np.zeros(pano_shape)
    #result = np.zeros(np.shape(src_img))


    # segment out images.
    n_seg = np.max(segmentsL) + 1
    seg_images = [[]] * n_seg
    shifts = [[]] * n_seg
    transformed_imgs = [[]] * n_seg
    x_dim = 0
    y_dim = 0
    max_shift = [0,0]

    for i in range(n_seg):
        seg_mask = (segmentsL == i).astype('uint8')
        seg_images[i] = src_img * np.repeat(seg_mask[:,:,np.newaxis],3,axis=2)
    
        # Apply homographies
        result1,transformed_imgs[i], shifts[i] = stitch.shiftImage(seg_images[i], dst_img, H_list[i])
        if np.max(shifts[i]) > 1920:
            result1,transformed_imgs[i], shifts[i] = stitch.shiftImage(seg_images[i], dst_img, H_list[0])
        #result1, result2, mask1, mask2 = stitch.applyHomography(dst_img, seg_images[i] , H_list[i])
        trans_shape = np.shape(transformed_imgs[i])

        y_tmp = max(trans_shape[0] + shifts[i][0], shifts[i][0] + np.shape(result1)[0])
        x_tmp = max(trans_shape[1] + shifts[i][1], shifts[i][0] + np.shape(result1)[1]) 
        
        if trans_shape[0] > y_dim:
        	y_dim = trans_shape[0]
        if trans_shape[1] > x_dim:
        	x_dim = trans_shape[1]
        if shifts[i][0] > max_shift[0]:
        	max_shift[0] = shifts[i][0]
        if shifts[i][1] > max_shift[1]:
        	max_shift[1] = shifts[i][1]

    final_pano = np.zeros((y_dim + max_shift[0], x_dim+max_shift[1], 3)).astype('float')
    for i in range(n_seg):
    	trans_shape = np.shape(transformed_imgs[i])
    	min_idx = [max_shift[0] - shifts[i][0], max_shift[1] - shifts[i][1]]
    	max_idx = [min_idx[0] + trans_shape[0], min_idx[1] + trans_shape[1]]

        final_pano[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], :] = final_pano[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], :] + transformed_imgs[i]/2
    
    final_pano[min_idx[0]:min_idx[0] + np.shape(result1)[0], min_idx[1]:min_idx[1] + np.shape(result1)[1]] = final_pano[min_idx[0]:min_idx[0] + np.shape(result1)[0], min_idx[1]:min_idx[1] + np.shape(result1)[1]] + result1/2 
    print("Final Pano Shape: " + str(final_pano.shape))


    transformed_imgs.insert(0,dst_img)
    shifts.insert(0,(0,0))

    for i in range(n_seg+1):
    	pre_pad = [max_shift[0] - shifts[i][0], max_shift[1] - shifts[i][1]]
    	post_pad = [final_pano.shape[0] - (pre_pad[0] + transformed_imgs[i].shape[0]), final_pano.shape[1] - (pre_pad[1] + transformed_imgs[i].shape[1])]
    	#print(pre_pad,post_pad,transformed_imgs[i].shape)
    	transformed_imgs[i] = np.pad(transformed_imgs[i],((pre_pad[0],post_pad[0]),(pre_pad[1],post_pad[1]),(0,0)),'constant',constant_values=(0,0))
    	print(transformed_imgs[i].shape)
    	#cv2.imshow("trans",transformed_imgs[i])
    	#cv2.waitKey(0)

    
    #final_pano = transformed_imgs[0]/2 + transformed_imgs[1]/2
    t = time.time()
    final_pano = performBlending2(transformed_imgs)
    print("Blending Time: "  + str(time.time() - t))

    return final_pano.astype('uint8')

def fit_plane(X):
    n_points = float(np.shape(X)[1])
    c = 1/n_points * np.sum(X,axis=1)
    A = (X.T - c).T
    U,S,V = np.linalg.svd(A)

    n = U[:,2]

    return n,c

def meanSquaredDistance(X, n,c):
    #N = float(np.shape(X)[1])
    distance = np.dot(n.T, (X.T - c).T)
    #msd = 1/N * np.sum(np.square(distance))
    msd = np.sum(np.square(distance))
    return msd

def computeHError(H_list, kp_list_L, kpsL, kpsR):
    # Computes the Mean Squared Error of the alignment of X using H. 
    err = 0
    num_points = 0

    for i in range(len(H_list)):
        # Compute necessary 
        # format X into homogenous coordinates
        points = kpsL[kp_list_L[i][0],:]
        x_len = len(kp_list_L[i][0])
        X = np.concatenate((points, np.ones((x_len,1))), axis=1).T
        y = np.concatenate((kpsR[kp_list_L[i][1],:],np.ones((x_len,1))),axis=1).T

        #Increment used number of points
        num_points += x_len

        # Multiply X and H
        y_bar = np.dot(H_list[i],X)
        y_bar = y_bar / y_bar[2,:]

        #Adjust SSE
        tmp = np.power(y_bar - y, 2)
        err += np.sum(np.sum(tmp))

    # Divide by number of points used.
    err = err / num_points

    return err 

def main(l_img, r_img,segmentsL, frame_id, SHOW_PANO=False, SAVE_TXT=True, SAVE_SINGLE = False):

    # Crop out padding
    l_img = l_img[50:-50,330:-330,:]
    r_img = r_img[50:-50,330:-330,:]
    # Calibration parameters provided by MICCAI 2017
    W = 1280                                                      # width of the images in pixels
    H = 1024                                                      # height of the images in pixels
    Camera_L_F = np.array([1068.39, 1068.19])                               # left camera x,y focal dist in pixels
    Camera_L_C = np.array([600.90, 500.74])                                 # left camera x,y center in pixels
    Camera_L_Alpha = 0.00000                                      # left camera skew
    Camera_L_K = np.array([-0.00087, 0.00238, 0.00012, 0.00000, 0.00000])   # left camera radial distortion
    Camera_R_F = np.array([1067.83, 1067.60])                               # right camera x,y focal dist in pixels
    Camera_R_C = np.array([698.22, 501.65])                                 # right camera x,y center in pixels
    Camera_R_Alpha = 0.00000                                                # right camera skew
    Camera_R_K = np.array([-0.00158, 0.00403, 0.00009, 0.00000, 0.00000])   # right camera radial distortion
    Extrinsic_Omega = np.array([-0.0001, -0.0013, 0.0000])                  # left to right camera rotation
    Extrinsic_T = np.array([-4.2773, -0.0440, -0.0303])                     # left to right camera position

    # Convert Camera parameters from provided values to OpenCV desired coordinates. 
    R, _ = cv2.Rodrigues(Extrinsic_Omega)
    K = np.eye(4)
    K[0:3,0:3] = R
    K[0:3,3] = Extrinsic_T.T
    CL = np.mat([[Camera_L_F[0],Camera_L_Alpha,Camera_L_C[0]],[0,Camera_L_F[1],Camera_L_C[1]],[0,0,1]])
    CR = np.mat([[Camera_R_F[0],Camera_R_Alpha,Camera_R_C[0]],[0,Camera_R_F[1],Camera_R_C[1]],[0,0,1]])
    P = CL * K[0:3,:]

    # Using the identity instead of camera intrinsics because camera intrinsics don't seem to work very well. 
    CL = np.eye(3)
    CR = np.eye(3)

    # Perform feature matching
    kpsR, desR = stitch.detectAndDescribe(r_img)
    kpsL, desL = stitch.detectAndDescribe(l_img)

    (matches, H, status) = stitch.matchKeypoints(kpsR,kpsL,desR,desL,.75,4.0)
    vis = stitch.drawMatches(r_img, l_img, kpsR, kpsL, matches, status)

    # Gather depth map of features
    idr = []
    idl =  []
    for k in range(len(matches)):
        idr.append(matches[k][1])
        idl.append(matches[k][0])

    points3D = cv2.triangulatePoints(np.dot(CR, np.eye(4)[0:3,:]), np.dot(CL, K[0:3,:]), kpsR[idr].T, kpsL[idl].T)
    points3D /= points3D[3]

    #cv2.imshow("Matches", vis)
    #cv2.waitKey(0)

    # match points to segments
    #seg_list_R = np.unique(segmentsR)
    seg_list_L = np.unique(segmentsL)

    #kp_list_R = [[]] * (np.max(segmentsR) + 1)
    #kp_list_L = [[[],[]]] * (np.max(segmentsL) + 1)

    kp_list_L = []
    seg3D_id = []

    for k in range(np.max(segmentsL) + 1):
        kp_list_L.append([[],[]])
        seg3D_id.append([])

    for k in range(len(matches)):
        r_point = kpsR[matches[k][1]].astype('int')
        #r_seg = segmentsR[r_point[1],r_point[0]]

        l_point = kpsL[matches[k][0]].astype('int')                                                                
        l_seg = segmentsL[l_point[1],l_point[0]]
        #print(l_seg)
        #print("Matches " + str(k) +": " + str(matches[k][0]))
        kp_list_L[l_seg][0].append(matches[k][0])
        kp_list_L[l_seg][1].append(matches[k][1])
        seg3D_id[l_seg].append(k)


    ## Perform planar based clustering.

    #points3D = points3D

    #Check reprojection Error
    #xL = np.dot(np.dot(CL, K[0:3,:]), points3D)
    #xR = np.dot(np.dot(CR, np.eye(4)[0:3,:]), points3D)

    #print(np.linalg.norm(xL[0:2,:] - kpsL[idr].T)/np.shape(xL)[1])
    #print(np.linalg.norm(xR[0:2,:] - kpsR[idr].T)/np.shape(xR)[1])

    # Fit planes to features
    n,c = fit_plane(points3D[0:3,:])
    sse_single = meanSquaredDistance(points3D[0:3,:],n,c)/len(points3D[0,:])


    # Compute planar error for multiplanar case
    sse_multi = 0
    n_points = 0
    for i in range(len(kp_list_L)):
        if (len(kp_list_L[i][0]) > 0):
            n,c = fit_plane(points3D[0:3,seg3D_id[i]])
            tmp = meanSquaredDistance(points3D[0:3,seg3D_id[i]],n,c)
            sse_multi = sse_multi + tmp
            n_points += len(points3D[0,:])

    sse_multi = sse_multi / n_points


    print("Single planar SSE: " + str(sse_single))
    print("Multiplanar SSE: " + str(sse_multi))
    print("Improvement: " +str( (1 - sse_multi/sse_single) * 100) + "%")

    t = time.time()        
    # Compute Homographies
    H_list = [[]] * len(kp_list_L)
    for i in range(len(kp_list_L)):
    	#print(kpsL[kp_list_L[i][0],:])
    	if (len(kp_list_L[i][0]) > 8):
            (H_list[i], status) = cv2.findHomography(kpsL[kp_list_L[i][0],:], kpsR[kp_list_L[i][1],:], cv2.RANSAC,4)
        
        if (len(H_list[i]) < 2):
        	H_list[i] = H_list[0]

    pano_multi = blend_images(l_img, r_img, segmentsL, H_list)
    mp_time = time.time() - t

    t = time.time()
    # Compare to panorama constructed using only background
    result1, result2, mask1, mask2 = stitch.applyHomography(l_img, r_img, H_list[0])
    pano_single = 0.5 * result1 * mask1 + 0.5 * result2 * mask2
    sp_time = time.time() - t

    # Compute MSE of alignment
    sp_points = [[[],[]]]
    for i in range(len(matches)):
        sp_points[0][0].append(matches[i][0])
        sp_points[0][1].append(matches[i][1])

    
    sp_err = computeHError([H], sp_points ,kpsL, kpsR)
    mp_err = computeHError(H_list, kp_list_L, kpsL, kpsR )
    print("Single planar H error: " + str(sp_err))
    print("Multi Planar H Error:  " + str(mp_err))

    # Adjust output size to remove black outer edges.
    y_vals, x_vals,_ = np.nonzero(pano_multi)
    pano_multi = pano_multi[np.min(y_vals):np.max(y_vals)+1,np.min(x_vals):np.max(x_vals)+1, :]

    if SHOW_PANO:
        ## Construct panorama
        b,g,r = cv2.split(pano_multi)
        pano_multi_plt = cv2.merge([r,g,b])
        plt.imshow(pano_multi_plt)

 
        b,g,r = cv2.split(pano_single)       # get b,g,r
        pano_single_plt = cv2.merge([r,g,b])
        plt.figure()
        plt.imshow(pano_single_plt.astype('uint8'))
        plt.show()
    
    if SAVE_TXT:
        with open(r'output.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([frame_id,sse_single, sse_multi, sp_err, mp_err, sp_time, mp_time])

    if SAVE_SINGLE: 
    	cv2.imwrite('./output/single_pano.jpg', pano_single)


    return pano_multi

if __name__ == "__main__":

    #n = np.mat([1,1,1]).T
    #c = np.mat([1,1,1])
    # 
    #x = np.random.rand(5)
    #y = np.random.rand(5)
    #z = c[0,2] - (n[0,0]*(x - c[0,0]) + n[1,0]*(y - c[0,1]))/n[2,0]
    #z[0] = z[0] + 5
    #points = np.array([x,y,z])
    #out = meanSquaredDistance(points,n,c)
    #out1 = meanSquaredDistance(points[:,0:2],n,c)
    #ut2 = meanSquaredDistance(points[:,2:],n,c)

    #print(out)
    #print(out1 + out2)
    #exit()

    GT_SEG = False
    # Initialize csv file
    if not path.exists('output.csv'):
        fields = ["Frame ID", "SP Plane Error", "MP Plane Error", "SP Stitch Error", "MP Stitch Error", "SP Timing","MP Timing"]
        with open(r'output.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    # Load images from directoryd
    im_dir = './instrument_dataset_6/'
    id_min = 1
    id_max = 1
    
    for im_idx in range(id_min,id_max+1):
        print("Processing Image " + str(im_idx) + "...")

        l_img = cv2.imread(im_dir + "left_frames/frame" + str(im_idx).zfill(3) + '.png') # Segmented source image
        r_img = cv2.imread(im_dir + "right_frames/frame" + str(im_idx).zfill(3) + '.png') # destination image

        if GT_SEG:
            #left_label = cv2.imread(im_dir + "ground_truth/Maryland_Bipolar_Forceps_labels/frame" + str(im_idx).zfill(3) + '.png')
            #right_label = cv2.imread(im_dir + "ground_truth/Right_Prograsp_Forceps_labels/frame" + str(im_idx).zfill(3) + '.png')
            #other_label = cv2.imread(im_dir + "ground_truth/Other_labels/frame" + str(im_idx).zfill(3) + '.png')
            
            left_label = cv2.imread(im_dir + "ground_truth/Bipolar_Forceps_labels/frame" + str(im_idx).zfill(3) + '.png')
            right_label = cv2.imread(im_dir + "ground_truth/Grasping_Retractor_labels/frame" + str(im_idx).zfill(3) + '.png')
            other_label = cv2.imread(im_dir + "ground_truth/Vessel_Sealer_labels/frame" + str(im_idx).zfill(3) + '.png')


            segmentsL = combineSegments([left_label,right_label,other_label]).astype('uint8')
            #print(np.shape(segmentsL))
            #print(np.shape(left_label + right_label + other_label))
            #segmentsL = left_label + right_label + other_label
            segmentsL = segmentsL[:,:,0]
            segmentsL = segmentsL[50:-50,330:-330]
        else:
            seg_dirs = os.listdir(im_dir+"ground_truth/")
            label_list = [[]] * len(seg_dirs)
            for i in range(len(seg_dirs)):
            	label_list[i] = cv2.imread(im_dir + "ground_truth/" + seg_dirs[i] + "/frame" + str(im_idx).zfill(3) + '.png')
            segmentsL = combineSegments(label_list).astype('uint8')
            print(np.unique(segmentsL))
            segmentsL = segmentsL[:,:,0]
            segmentsL = segmentsL[50:-50,330:-330]
           

        # Images from the MICCAI dataset have black padding, so we crop them to get rid of that. 
        #cv2.imshow("left",l_img)
        #cv2.imshow('right',r_img)

        out_pano = main(l_img, r_img,segmentsL, im_idx, SAVE_TXT=True)
        out_file = './output/pano_' + str(im_idx) + '.jpg'
        cv2.imwrite(out_file, out_pano)