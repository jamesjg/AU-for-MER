import numpy as np
import pandas as pd
import cv2
import numpy as np
import pandas as pd
import cv2
def minmax_norm(x):
    x_flat = x.reshape(-1)
    x_max = np.max(x)
    x_min = np.min(x)
    if x_max == x_min:
        x_flat *= 0
    else:
        x_flat = (x_flat - x_min) / (x_max - x_min)
    return x_flat.reshape(x.shape)


def calc_os_flow(flow):
    u_flow = minmax_norm(flow[:, :, 0])*255
    v_flow = minmax_norm(flow[:, :, 1])*255
    
    ux, uy = np.gradient(flow[:, :, 0])
    vx, vy = np.gradient(flow[:, :, 1])
    
    os_flow = np.sqrt(ux ** 2 + vy ** 2 + 0.25 * (uy + vx) ** 2)
    os_flow = minmax_norm(os_flow)*255
    
    return np.concatenate((os_flow.reshape(*os_flow.shape, 1), v_flow.reshape(*v_flow.shape, 1), u_flow.reshape(*u_flow.shape, 1)), axis=2)

def pol2cart(rho, phi): #Convert polar coordinates to cartesian coordinates for computation of optical strain
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def computeStrain(u, v):
    u_x= u - pd.DataFrame(u).shift(-1, axis=1)
    v_y= v - pd.DataFrame(v).shift(-1, axis=0)
    u_y= u - pd.DataFrame(u).shift(-1, axis=0)
    v_x= v - pd.DataFrame(v).shift(-1, axis=1)
    os = np.array(np.sqrt(u_x**2 + v_y**2 + 1/2 * (u_y+v_x)**2).ffill(1).ffill(0))
    return os

def resize(img, target_size=(28,28)):
    rimg = cv2.resize(img, (*target_size,), interpolation=cv2.INTER_LINEAR)
    return rimg

def perChannelNormalize(img):
    res = img.copy()
    for chnl in range(img.shape[2]):
        channel = res[:,:,chnl]
        res[:,:,chnl] = 255.*(channel - channel.min())/(channel.max() - channel.min())
    return res.astype(np.uint8)

def get_optical_flow(img1, img2):
    #Compute Optical Flow Features
    assert img1.shape == img2.shape
    # optical_flow = cv2.DualTVL1OpticalFlow_create() #Depends on cv2 version
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create(
        tau = 0.25, 
        #lambda_ = 0.15,
        theta = 0.3,
        nscales = 5, 
        warps=5,
        epsilon=0.01, 
        innnerIterations=30,
        outerIterations=10, 
        scaleStep=0.5,
        gamma=0.1,
        medianFiltering=5,
        useInitialFlow=False
    )
    flow = optical_flow.calc(img1, img2, None)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
    u, v = pol2cart(magnitude, angle)
    os = computeStrain(u, v)
                
    #Features Concatenation
    final = np.zeros((*img1.shape[0:2],3))
    final[:,:,0] = os #B
    final[:,:,1] = v #G
    final[:,:,2] = u #R

    # final = calc_os_flow(flow)
    return final

if __name__ == "__main__":
    from facenet_pytorch import MTCNN
    # path1 = "datasets/CASME2/CASME2_RAW_selected/sub01/EP02_01f/img46.jpg"
    # path2 = "datasets/CASME2/CASME2_RAW_selected/sub01/EP02_01f/img59.jpg"
    # path1 = "datasets/CASME3/Part_A_ME_clip/frame/spNO.1_b_166/166.jpg"
    # path2 = "datasets/CASME3/Part_A_ME_clip/frame/spNO.1_b_166/175.jpg"
    # path1 = "datasets/CASME2/Cropped/sub01/EP02_01f/reg_img46.jpg"
    # path2 = "datasets/CASME2/Cropped/sub01/EP02_01f/reg_img59.jpg"
    path1 = "datasets/CASME2/Cropped/sub02/EP01_11f/reg_img46.jpg"
    path2 = "datasets/CASME2/Cropped/sub02/EP01_11f/reg_img91.jpg"
    # img1_gray = cv2.imread(path1,0)
    # img2_gray = cv2.imread(path2,0)
    import dlib
    predictor_model = "shape_predictor_68_face_landmarks.dat"
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    side = max(img1.shape[0], img1.shape[1])
    # mtcnn = MTCNN(margin=0, image_size=224, select_largest=True, post_process=False, device='cuda:0')
    # box1, score1, points1 = mtcnn.detect(img1, landmarks=True)
    # box2, score2, points2 = mtcnn.detect(img2, landmarks=True)
 
    # # print(box1, points1)
    # # img1 = mtcnn(img1).permute(1,2,0).int().numpy().astype('uint8')
    # # img2 = mtcnn(img2).permute(1,2,0).int().numpy().astype('uint8')
    # box1, box2 = box1[0].astype(int), box2[0].astype(int)
    # points1, points2 = points1[0], points2[0]
    # eye_dis1 = np.linalg.norm(points1[0] - points1[1])*5/12
    # eye_dis2 = np.linalg.norm(points2[0] - points2[1])*5/12
    # box1[0] = int(max(0, points1[0][0] - eye_dis1*0.9))
    # box1[1] = int(max(0, points1[0][1] - eye_dis1*1.4))
    # box1[2] = int(min(img1.shape[1], points1[1][0] + eye_dis1*0.9))
    # box2[0] = int(max(0, points2[0][0] - eye_dis2*0.9))
    # box2[1] = int(max(0, points2[0][1] - eye_dis2*1.4))
    # box2[2] = int(min(img2.shape[1], points2[1][0] + eye_dis2*0.9))
    # img1 = img1[box1[1]:box1[3], box1[0]:box1[2]]
    # img2 = img2[box2[1]:box2[3], box2[0]:box2[2]]
    # img1 = cv2.resize(img1, (224, 224), interpolation=cv2.INTER_LINEAR)
    # img2 = cv2.resize(img2, (224, 224), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("img1.png", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    cv2.imwrite("img2.png", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # import ipdb; ipdb.set_trace()
    # img1 = cv2.resize(img1, (side, side), interpolation=cv2.INTER_LINEAR)
    # img2 = cv2.resize(img2, (side, side), interpolation=cv2.INTER_LINEAR)
    # img1_gray = img1_gray[box1[1]:box1[3], box1[0]:box1[2]]
    # img2_gray = img2_gray[box2[1]:box2[3], box2[0]:box2[2]]
    # img1_gray = cv2.resize(img1_gray, (side, side), interpolation=cv2.INTER_LINEAR)
    # img2_gray = cv2.resize(img2_gray, (side, side), interpolation=cv2.INTER_LINEAR)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # img1_gray = cv2.resize(img1_gray, (128, 128), interpolation=cv2.INTER_LINEAR)
    # img2_gray = cv2.resize(img2_gray, (128, 128), interpolation=cv2.INTER_LINEAR)
    detect = face_detector(img1, 1)
    shape = face_pose_predictor(img1, detect[0])
    #Left Eye
    x11=max(shape.part(36).x - 15, 0)
    y11=shape.part(36).y 
    x12=shape.part(37).x 
    y12=max(shape.part(37).y - 15, 0)
    x13=shape.part(38).x 
    y13=max(shape.part(38).y - 15, 0)
    x14=min(shape.part(39).x + 15, 128)
    y14=shape.part(39).y 
    x15=shape.part(40).x 
    y15=min(shape.part(40).y + 15, 128)
    x16=shape.part(41).x 
    y16=min(shape.part(41).y + 15, 128)
    
    #Right Eye
    x21=max(shape.part(42).x - 15, 0)
    y21=shape.part(42).y 
    x22=shape.part(43).x 
    y22=max(shape.part(43).y - 15, 0)
    x23=shape.part(44).x 
    y23=max(shape.part(44).y - 15, 0)
    x24=min(shape.part(45).x + 15, 128)
    y24=shape.part(45).y 
    x25=shape.part(46).x 
    y25=min(shape.part(46).y + 15, 128)
    x26=shape.part(47).x 
    y26=min(shape.part(47).y + 15, 128)
    
    #ROI 1 (Left Eyebrow)
    x31=max(shape.part(17).x - 12, 0)
    y32=max(shape.part(19).y - 12, 0)
    x33=min(shape.part(21).x + 12, 128)
    y34=min(shape.part(41).y + 12, 128)
    
    #ROI 2 (Right Eyebrow)
    x41=max(shape.part(22).x - 12, 0)
    y42=max(shape.part(24).y - 12, 0)
    x43=min(shape.part(26).x + 12, 128)
    y44=min(shape.part(46).y + 12, 128)
    
    #ROI 3 #Mouth
    x51=max(shape.part(60).x - 12, 0)
    y52=max(shape.part(50).y - 12, 0)
    x53=min(shape.part(64).x + 12, 128)
    y54=min(shape.part(57).y + 12, 128)
    
    #Nose landmark
    x61=shape.part(28).x
    y61=shape.part(28).y

    print(img1_gray.shape)
    opfl = get_optical_flow(img1_gray, img2_gray) # get (u,v,os)
        #Remove global head movement by minus nose region
    # opfl = opfl[:, :, ::-1] # BGR to RGB
    # # opfl[:, :, 0] = abs(opfl[:, :, 0] - opfl[y61-5:y61+6, x61-5:x61+6, 0].mean())
    # # opfl[:, :, 1] = abs(opfl[:, :, 1] - opfl[y61-5:y61+6, x61-5:x61+6, 1].mean())
    # # opfl[:, :, 2] = opfl[:, :, 2] - opfl[y61-5:y61+6, x61-5:x61+6, 2].mean()
    # opfl = opfl[:, :, ::-1] # RGB to BGR
    opfl = resize(opfl) # resize to 28x28
    opfl = perChannelNormalize(opfl) # normalize each channel to 0..255 
    cv2.imwrite("opfl.png", opfl) # save the image