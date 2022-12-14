import os
import shutil
import cv2

video_dir = '/data1/alexhu/smplify-x-master-video/smplifyx/output/images'
image_list = sorted(os.listdir(video_dir))

fps = 30
size = (1000, 720)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
ori_video = cv2.VideoWriter('/data1/alexhu/smplify-x-master-video/smplifyx/output/ori.mp4', fourcc, fps, size)
video = cv2.VideoWriter('/data1/alexhu/smplify-x-master-video/smplifyx/output/out.mp4', fourcc, fps, size)
expose_video = cv2.VideoWriter('/data1/alexhu/smplify-x-master-video/smplifyx/output/expose_out.mp4', fourcc, fps, size)


for img_name in image_list:
    real_ori_img_path = os.path.join('/data1/alexhu/smplify-x-master-video/smplifyx/data_folder/images', img_name+'.jpg')
    real_img_path = os.path.join(video_dir, img_name, '000', 'output.png')
    out_img_path = os.path.join('/data1/alexhu/smplify-x-master-video/smplifyx/output', 'merge', img_name+'.png')
    shutil.copy(real_img_path, out_img_path)

    img = cv2.imread(real_img_path)
    video.write(img)
    img_ori = cv2.imread(real_ori_img_path)
    ori_video.write(img_ori)
    img_expose = cv2.imread(os.path.join('/data1/alexhu/Datasets/expose_test_1/NMFs_CSL/545/expose','hd_overlay_'+img_name+'.png'))
    expose_video.write(img_expose)

video.release()
ori_video.release()
expose_video.release()