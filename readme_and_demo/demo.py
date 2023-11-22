import os
import pickle as pkl
import numpy as np
import render_model
from smpl.smpl_webuser.serialization import load_model
import cv2
import warnings

def renderImage(model,img_path,camPose,camIntrinsics):


    img = cv2.imread(img_path)
    class cam:
        pass
    cam.rt = cv2.Rodrigues(camPose[0:3,0:3])[0].ravel()
    cam.t = camPose[0:3,3]
    cam.f = np.array([camIntrinsics[0,0],camIntrinsics[1,1]])
    cam.c = camIntrinsics[0:2,2]
    h = int(2*cam.c[1])
    w = int(2*cam.c[0])
    im = (render_model.render_model(model, model.f, w, h, cam, img= img)* 255.).astype('uint8')
    return im

if __name__ == '__main__':

    # set the path and open the pose model
    seq_name = 'courtyard_basketball_00'
    seq_datasetDir = r'/home/chenpc/exp_3DPW/sequenceFiles/sequenceFiles/train'
    img_datasetDir = r'/home/chenpc/exp_3DPW/imageFiles'
    file = os.path.join(seq_datasetDir,seq_name+'.pkl')
    seq = pkl.load(open(file,'rb'), encoding='latin1')
    print(seq['v_template_clothed'])

    # depend on the gender to open the smpl model
    warnings.filterwarnings("ignore",category=DeprecationWarning)       # ignore the warning: DeprecationWarning: Please use `csc_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.csc` namespace is deprecated.
    models = list()
    for iModel in range(0,len(seq['v_template_clothed'])):
        if seq['genders'][iModel] == 'm':
            with open(r'/home/chenpc/exp_3DPW/3DPW_ENV/lib/python3.10/site-packages/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl', 'rb') as f:
                model = load_model(pkl.load(f, encoding='latin1'))
                print("male")
        else:
            with open(r'/home/chenpc/exp_3DPW/3DPW_ENV/lib/python3.10/site-packages/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl', 'rb') as f:
                model = load_model(pkl.load(f, encoding='latin1'))
                print("female")

        model.betas[:10] = seq['betas'][iModel][:10]
        models.append(model)
        print(models)
        print("0")

    # depend on the sequence to generate the picture with photo and 3D model
    iModel = 0
    iFrame = 25
    if seq['campose_valid'][iModel][iFrame]:
        print("compose_valid:", seq['campose_valid'])
        # print("poses:", seq['poses'])
        models[iModel].pose[:] = seq['poses'][iModel][iFrame]
        models[iModel].trans[:] = seq['trans'][iModel][iFrame]
        img_path = os.path.join(img_datasetDir,seq['sequence']+'/image_{:05d}.jpg'.format(iFrame))
        im = renderImage(models[iModel],img_path,seq['cam_poses'][iFrame],seq['cam_intrinsics'])
        cv2.imshow('3DPW Example',im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print('end')