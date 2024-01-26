import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os

def project(a, f=8.445e-04, m=173913.04, cx=3.200e+02, cy=2.400e+02, h=0.014):
    a[2] += h
    b = np.array([[0., 0., 1.]])
    inner_product = (a * b).sum()
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = inner_product / (a_norm * b_norm)

    theta = np.arccos(cos)
    omega = np.arctan2(a[1],a[0]) + np.pi
    # print("proj omega", omega)
    # print("proj theta", theta)
    r = m * f * theta

    p = np.zeros((2))
    p[0] = r * np.cos(omega) + cx
    p[1] = r * np.sin(omega) + cy

    return p

def project_inverse(p, f=8.445e-04, m=173913.04,
                    cx=3.200e+02, cy=2.400e+02, r=0.015, h=0.014):
    p1 = p - torch.tensor([cx, cy])
    # print("p1", p1[0])
    # p1 = p
    omega = torch.atan2(p1[:, 1], p1[:, 0])
    # print("inv omega", omega[0])
    theta = torch.norm(p1, dim=1) / (m*f)
    # print("inv theta", theta[0])
    x1 = -torch.cos(omega) * torch.sin(theta)
    y1 = -torch.sin(omega) * torch.sin(theta)
    z1 = torch.cos(theta)

    k = (h*z1 + (-h**2*x1**2 - h**2*y1**2 + r**2*x1**2 + r**2*y1**2 + r**2*z1 **2)**0.5)
    a = k.unsqueeze(-1) * torch.stack([x1, y1, z1], dim=1)
    a[:, 2] -= h
    return a

rest_pos = None
last_pos = None

# for i in range(215, 400):
#     idx = i
#     path = "../data/gelsight-force-capture-press-twist-x_2023-06-27-16-48-28/frame%05d.png" % idx
#
#     save_path = "../data/blob_data_rot_x"
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     img = cv2.imread(path, 0)
#     curve1 = 50
#     curve2 = 100
#     mask = img<curve1
#     img1 = (curve2/curve1)*img
#     img2 = (255-(255-curve2)/(255-curve1)*(255-img))
#     img = img1*mask + img2*(1-mask)
#     img = img.astype('uint8')
#
#     params = cv2.SimpleBlobDetector_Params()
#     params.minThreshold = 0
#     params.minThreshold = 0
#
#     detector = cv2.SimpleBlobDetector_create(params)
#     keypoints = detector.detect(img)
#
#     # img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
#     #                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#     # cv2.imshow("Blobs", img_with_keypoints)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     def step_pos(last_pos, pos):
#         d = torch.cdist(last_pos, pos)
#         thres = 5
#         mask = d.min(dim=1)[0]<thres
#         idx = d.argmin(dim=1)
#         match_pos = pos[idx]
#         match_pos[mask.logical_not()] = last_pos[mask.logical_not()]
#         return match_pos
#
#     pos = torch.tensor([pt.pt for pt in keypoints])
#
#     if rest_pos is None:
#         mask = torch.logical_and(pos[:, 0]>260, pos[:, 0]<390)
#         rest_pos = pos[mask]
#         last_pos = pos[mask]
#         pos_3d = project_inverse(rest_pos)
#         # for j in range(1):
#         #     pos_now = project(pos_3d[j].detach().cpu().numpy())
#         #     print(pos_now, rest_pos[j])
#         # fig = plt.figure()
#         # ax = fig.add_subplot(projection='3d')
#         # ax.set_xlim(-0.01, 0.01)
#         # ax.set_ylim(-0.01, 0.01)
#         # ax.set_zlim(-0.01, 0.01)
#         # ax.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2])
#         np.save(os.path.join(save_path, "pos_ori_3d.npy"), pos_3d.detach().cpu().numpy())
#         # np.save(os.path.join(save_path, "pos_ori_2d.npy"), rest_pos.detach().cpu().numpy())
#
#         # for ww in range(pos_3d.shape[0]):
#         #     print(pos_3d[ww], torch.norm(pos_3d[ww]))
#         # plt.show()
#
#     pos = step_pos(last_pos, pos)
#     pos_3d = project_inverse(pos)
#     np.save(os.path.join(save_path, f"pos_3d_{i}.npy"), pos_3d.detach().cpu().numpy())
#     # print(pos.shape)
#     delta = pos-rest_pos
#
#     last_pos = pos
#
#     # plt.scatter(rest_pos[:, 0], rest_pos[:, 1])
#     # plt.scatter(pos[:, 0], pos[:, 1])
#     # plt.savefig(os.path.join(save_path, 'blob_tmp/%i.jpg'%i))
#     # plt.close()
#     print(i)