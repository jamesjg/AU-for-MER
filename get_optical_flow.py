import cv2
import numpy as np



def tvl1_ofcalc(path1, path2):
    origin_photo1=cv2.imread(path1)
    resized_photo1=cv2.resize(origin_photo1,(224,224))
    origin_photo2=cv2.imread(path2)
    resized_photo2=cv2.resize(origin_photo2,(224,224))
    img1 = cv2.cvtColor(resized_photo1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(resized_photo2, cv2.COLOR_BGR2GRAY)

    flow = cv2.optflow.createOptFlow_DualTVL1()
    # img1 = cv2.resize(img1, (28, 28))
    # img2 = cv2.resize(img2, (28, 28))
    # flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # u = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    # v = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
    # magnitude = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # image_u_v_os_temp = np.zeros([28, 28, 3],dtype=np.uint8)
    # image_u_v_os_temp[:, :, 0] = u
    # image_u_v_os_temp[:, :, 1] = v
    # image_u_v_os_temp[:, :, 2] = magnitude
    # return image_u_v_os_temp
    # flow = cv2.optflow.DualTVL1OpticalFlow_create()
    # flow = cv2.DualTVL1OpticalFlow_create()
    # flow.setEpsilon(epsilon)  # 设置epsilon参数的值
    of = flow.calc(img1, img2, None)

    # flow = cv2.optflow.createOptFlow_DeepFlow()
    # of = flow.calc(img1, img2, None)

    # of = cv2.optflow.calcOpticalFlowSparseToDense(img1,img2)

    # flow = cv2.DISOpticalFlow_create(2)
    # of = flow.calc(img1, img2, None)
    return of

def minmax_norm(x):
    x_flat = x.reshape(-1)
    x_max = np.max(x)
    x_min = np.min(x)
    if x_max == x_min:
        x_flat *= 0
    else:
        x_flat = (x_flat - x_min) / (x_max - x_min)
    return x_flat.reshape(x.shape)


def calc_os_flow(path1, path2):
    flow = tvl1_ofcalc(path1, path2)
    u_flow = minmax_norm(flow[:, :, 0])*255
    v_flow = minmax_norm(flow[:, :, 1])*255
    
    ux, uy = np.gradient(flow[:, :, 0])
    vx, vy = np.gradient(flow[:, :, 1])
    
    os_flow = np.sqrt(ux ** 2 + vy ** 2 + 0.25 * (uy + vx) ** 2)
    os_flow = minmax_norm(os_flow)*255
    
    return np.concatenate((os_flow.reshape(*os_flow.shape, 1), v_flow.reshape(*v_flow.shape, 1), u_flow.reshape(*u_flow.shape, 1)), axis=2)


if __name__ == '__main__':
    path1 = "datasets/CASME2/scrfd_cropped_selected/sub01/EP02_01f/img46.jpg"
    path2 = "datasets/CASME2/scrfd_cropped_selected/sub01/EP02_01f/img59.jpg"

    path1 = "datasets/CASME2/CASME2_RAW_selected/sub01/EP02_01f/img46.jpg"
    path2 = "datasets/CASME2/CASME2_RAW_selected/sub01/EP02_01f/img59.jpg"
    os_flow = calc_os_flow(path1, path2)
    os_flow = cv2.resize(os_flow, (28, 28), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
    cv2.imwrite("os_flow.jpg", os_flow)
