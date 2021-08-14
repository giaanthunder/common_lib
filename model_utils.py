import os, sys, time, math, random

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)

from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import cv2
from PIL import Image, ImageOps, ImageSequence
import matplotlib.pyplot as plt

from utility import pwd, cd

cur_dir = os.path.abspath(os.path.dirname(__file__)) + '/'


# sys.path.append('/media/anhuynh/DATA/03_task/common_lib')


class FaceDetector():
    def __init__(self):
        retina_path = cur_dir + 'RetinaFace/'
        if retina_path not in sys.path:
            sys.path.append(retina_path)

        from RetinaFace.retinaface import RetinaFace
        self.detector = RetinaFace(retina_path+'model/R50', 0, 0, 'net3')

    def detect(self, img):
        target_size = 1024
        max_size = 1980
        im_size_min = np.min(img.shape[0:2])
        im_size_max = np.max(img.shape[0:2])

        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        try:
            bbox, landmark5 = self.detector.detect(img, threshold=0.8, scales=[im_scale])
        except:
            print('Detect error')
            print(img.shape)
            return np.array([]), np.array([]), np.array([])

        bboxes = []
        landmarks = []
        scores = []

        for i in range(bbox.shape[0]):
            x1, y1, x2, y2, score = bbox[i].astype(np.int)
            landmark = landmark5[i].astype(np.int)
            bboxes.append([x1, y1, x2, y2])
            landmarks.append(landmark)
            scores.append(score)

        bboxes = np.array(bboxes)
        landmarks = np.array(landmarks)
        scores = np.array(scores)

        return bboxes, landmarks, scores


class FaceDetector2():
    def __init__(self):
        retina_path = cur_dir + 'retinaface_tf2/'
        if retina_path not in sys.path:
            sys.path.append(retina_path)
        from retinaface_tf2.modules.models import RetinaFaceModel
        from retinaface_tf2.modules.utils import load_yaml

        cfg = load_yaml(retina_path+'configs/retinaface_res50.yaml')
    
        model = RetinaFaceModel(cfg, training=False, iou_th=0.4, score_th=0.5)
        checkpoint_dir = retina_path + 'checkpoints/' + cfg['sub_name']
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        self.model = model
        self.cfg   = cfg
        sys.path.remove(retina_path)

    def detect(self, img_raw):
        retina_path = cur_dir + 'retinaface_tf2/'
        if retina_path not in sys.path:
            sys.path.append(retina_path)
        from retinaface_tf2.modules.utils import pad_input_image, recover_pad_output
        
        max_size = 1980

        raw_h, raw_w, _ = img_raw.shape

        if raw_h > max_size:
            d = max_size/raw_h
            new_w = int(raw_w * d)
            img = cv2.resize(img_raw, (new_w,max_size), cv2.INTER_AREA)
        elif raw_w > max_size:
            d = max_size/raw_w
            new_h = int(raw_h * d)
            img = cv2.resize(img_raw, (max_size,new_h), cv2.INTER_AREA)
        else:
            img = img_raw


        img, pad_params = pad_input_image(img, max_steps=max(self.cfg['steps']))
        img = img.astype(np.float32)
        img = np.expand_dims(img,axis=0)
        outputs = self.model(img).numpy()
        outputs = recover_pad_output(outputs, pad_params)

        bboxes = []
        landmarks = []
        scores = []

        for output in outputs:
            # bbox
            x1 = int(output[0] * raw_w)
            y1 = int(output[1] * raw_h)
            x2 = int(output[2] * raw_w)
            y2 = int(output[3] * raw_h)
            bbox = np.array([x1,y1,x2,y2])
            bboxes.append(bbox)

            # confidence
            score = output[15]
            scores.append(score)

            # landmark
            l_eye_x   = int(output[ 4] * raw_w)
            l_eye_y   = int(output[ 5] * raw_h)
            r_eye_x   = int(output[ 6] * raw_w)
            r_eye_y   = int(output[ 7] * raw_h)
            nose_x    = int(output[ 8] * raw_w)
            nose_y    = int(output[ 9] * raw_h)
            l_mouth_x = int(output[10] * raw_w)
            l_mouth_y = int(output[11] * raw_h)
            r_mouth_x = int(output[12] * raw_w)
            r_mouth_y = int(output[13] * raw_h)
            landmark = np.array([
                [l_eye_x, l_eye_y], 
                [r_eye_x, r_eye_y], 
                [nose_x, nose_y], 
                [l_mouth_x, l_mouth_y], 
                [r_mouth_x, r_mouth_y]
            ])
            landmarks.append(landmark)

        bboxes = np.array(bboxes)
        landmarks = np.array(landmarks)
        scores = np.array(scores)

        return bboxes, landmarks, scores


class FaceParser():
    def __init__(self):
        import bisenet
        self.bnet = bisenet.models.pretrained_models()
        self.mask_dict = {
            'background': 0, 
            'skin' :  1, 'l_brow':  2, 'r_brow':  3, 'l_eye':  4, 'r_eye':  5,  'eye_g':  6, 
            'l_ear':  7, 'r_ear' :  8, 'ear_r' :  9, 'nose' : 10, 'mouth': 11,  'u_lip': 12, 
            'l_lip': 13, 'neck'  : 14, 'neck_l': 15, 'cloth': 16, 'hair' : 17,  'hat'  : 18
        }

    def parse(self, img, smooth=False, percent=10):
        import bisenet
        img = tf.convert_to_tensor(img)
        img_in = bisenet.data.preprocess(img, size=512)
        img_in = tf.expand_dims(img_in, axis=0)
        out, out16, out32 = self.bnet(img_in)
        label = out[0].numpy()
        masks = bisenet.data.to_mask2(img, label, smooth, percent)
        return masks

    def segment(self, img, masks, mask_names):
        black = np.zeros(img.shape, img.dtype)
        mask  = np.zeros(masks['skin'].shape, masks['skin'].dtype)
        for name in mask_names:
            mask += masks[name]
        seg_img = np.where(mask>0,img,black)
        return seg_img

    def blur_edge(self, mask, percent=10):
        import bisenet
        return bisenet.data.blur_edge(mask,percent)


class SuperResolution():
    def __init__(self):
        esrgan_path = cur_dir + 'esrgan_tf2/'
        if esrgan_path not in sys.path:
            sys.path.append(esrgan_path)
        from esrgan_tf2.modules.models import RRDB_Model
        from esrgan_tf2.modules.utils import load_yaml

        cfg = load_yaml(esrgan_path+'configs/esrgan.yaml')
        model = RRDB_Model(None, cfg['ch_size'], cfg['network_G'])
        checkpoint_dir = esrgan_path + 'checkpoints/' + cfg['sub_name']
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        self.model = model
        sys.path.remove(esrgan_path)

    # def x4(self,raw_img):
    def zoon(self,raw_img):
        esrgan_path = cur_dir + 'esrgan_tf2'
        if esrgan_path not in sys.path:
            sys.path.append(esrgan_path)
        from esrgan_tf2.modules.utils import tensor2img

        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
        raw_img = np.expand_dims(raw_img, axis=0)
        raw_img = raw_img / 255.
        try:
            sr_img = self.model(raw_img)
        except:
            print('Super resolution feedforward error')
            return None
        sr_img = tensor2img(sr_img)
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)
        return sr_img


class SuperResolution2():
    def __init__(self, scale=4):
        edsr_path = cur_dir + 'EDSR/src/'
        if edsr_path not in sys.path:
            sys.path.append(edsr_path)

        import torch
        import edsr_uti, model
        from option import args

        torch.manual_seed(args.seed)
        args.data_test = []
        checkpoint = edsr_uti.checkpoint(args)

        args.scale = [scale]
        self.args = args
        self.model = model.Model(args, checkpoint)
        checkpoint.done()

    def zoom(self, raw_img):
        import torch
        import edsr_uti
        torch.set_grad_enabled(False)

        self.model.eval()

        img = np.transpose(raw_img,[2,0,1])
        img = img.astype(np.float32)
        img = np.expand_dims(img,axis=0)
        gpu = torch.device("cuda:0")
        img = torch.from_numpy(img).float().to(gpu)

        sr = self.model(img, 0)
        sr = edsr_uti.quantize(sr, self.args.rgb_range)

        img = sr.cpu().numpy()[0].astype(np.uint8)
        img = np.transpose(img,[1,2,0])

        return img

class SuperResolution3():
    def __init__(self):
        src_path = cur_dir + 'Face-Super-Resolution/'
        if src_path not in sys.path:
            sys.path.append(src_path)
        a = pwd()
        cd(src_path)

        import torch
        import torchvision.transforms as transforms
        from models.SRGAN_model import SRGANModel
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu_ids', type=str, default=None)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--lr_G', type=float, default=1e-4)
        parser.add_argument('--weight_decay_G', type=float, default=0)
        parser.add_argument('--beta1_G', type=float, default=0.9)
        parser.add_argument('--beta2_G', type=float, default=0.99)
        parser.add_argument('--lr_D', type=float, default=1e-4)
        parser.add_argument('--weight_decay_D', type=float, default=0)
        parser.add_argument('--beta1_D', type=float, default=0.9)
        parser.add_argument('--beta2_D', type=float, default=0.99)
        parser.add_argument('--lr_scheme', type=str, default='MultiStepLR')
        parser.add_argument('--niter', type=int, default=100000)
        parser.add_argument('--warmup_iter', type=int, default=-1)
        parser.add_argument('--lr_steps', type=list, default=[50000])
        parser.add_argument('--lr_gamma', type=float, default=0.5)
        parser.add_argument('--pixel_criterion', type=str, default='l1')
        parser.add_argument('--pixel_weight', type=float, default=1e-2)
        parser.add_argument('--feature_criterion', type=str, default='l1')
        parser.add_argument('--feature_weight', type=float, default=1)
        parser.add_argument('--gan_type', type=str, default='ragan')
        parser.add_argument('--gan_weight', type=float, default=5e-3)
        parser.add_argument('--D_update_ratio', type=int, default=1)
        parser.add_argument('--D_init_iters', type=int, default=0)

        parser.add_argument('--print_freq', type=int, default=100)
        parser.add_argument('--val_freq', type=int, default=1000)
        parser.add_argument('--save_freq', type=int, default=10000)
        parser.add_argument('--crop_size', type=float, default=0.85)
        parser.add_argument('--lr_size', type=int, default=128)
        parser.add_argument('--hr_size', type=int, default=512)

        # network G
        parser.add_argument('--which_model_G', type=str, default='RRDBNet')
        parser.add_argument('--G_in_nc', type=int, default=3)
        parser.add_argument('--out_nc', type=int, default=3)
        parser.add_argument('--G_nf', type=int, default=64)
        parser.add_argument('--nb', type=int, default=16)

        # network D
        parser.add_argument('--which_model_D', type=str, default='discriminator_vgg_128')
        parser.add_argument('--D_in_nc', type=int, default=3)
        parser.add_argument('--D_nf', type=int, default=64)

        # data dir
        parser.add_argument('--pretrain_model_G', type=str, default='90000_G.pth')
        parser.add_argument('--pretrain_model_D', type=str, default=None)

        args = parser.parse_args()

        self.model = SRGANModel(args, is_train=False)
        self.model.load()

        self.trans = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
        cd(a)

    def zoom(self, raw_img):
        import torch
        input_img = cv2.resize(raw_img, (128,128), cv2.INTER_AREA)
        input_img = torch.unsqueeze(self.trans(input_img), 0)
        self.model.var_L = input_img.to(self.model.device)
        self.model.test()
        img = self.model.fake_H.squeeze(0).cpu().numpy()
        img = np.clip((np.transpose(img, (1, 2, 0)) / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        return img



class Classifier():
    def __init__(self, name, custom_proc=None, path=None):
        if path is not None:
            self.f_model = Extractor(name)
            self.c_model = tf.keras.models.load_model(path)
        else:
            self.f_model = Extractor(name, flip=True)
            self.c_model = create_model(self.f_model.f_len)
        self.custom_proc = custom_proc

    def process_data(self, pos_dir, neg_dir):
        pos_paths = ls(pos_dir)
        neg_paths = ls(neg_dir)

        x = []
        y = []
        for i in range(len(pos_paths)):
            print('%d/%d'%(i,len(pos_paths)))
            img = Image.open(pos_paths[i])
            img = np.array(img)
            if self.custom_proc is not None:
                img = self.custom_proc(img) 
            f1 = self.f_model.get_feature(img)
            x.append(f1)
            y.append([1])
            y.append([1])

        for i in range(len(neg_paths)):
            print('%d/%d'%(i,len(neg_paths)))
            img = Image.open(neg_paths[i])
            img = np.array(img)
            if self.custom_proc is not None:
                img = self.custom_proc(img) 
            f1 = self.f_model.get_feature(img)
            x.append(f1)
            y.append([0])
            y.append([0])

        x = np.concatenate(x,axis=0).astype(np.float32)
        y = np.concatenate(y,axis=0).astype(np.float32)

        return (x_npy, y_npy)

    def train(self, save_path, x_npy, y_npy):
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)

        x_train = x[idx]
        y_train = y[idx]

        self.c_model.fit(
            x_train, y_train,
            batch_size=4, epochs=20,
            # validation_split=0.2,
        )
        self.c_model.save(save_path)

    def test(self, img):
        if self.custom_proc is not None:
            img = self.custom_proc(img) 
        f1 = self.f_model.get_feature(img)
        score = self.c_model(f1)[0,1].numpy()
        return score


def create_model(f_len):
    inputs = tf.keras.Input(shape=(f_len,))
    x = layers.Dense(4096, activation=tf.nn.relu)(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(4096, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1024, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation=tf.nn.relu)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation=tf.nn.relu)(x)
    outputs = layers.Dense(2, activation=tf.nn.softmax)(x)
    c_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    c_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return c_model


def load_model(path):
    model = tf.keras.models.load_model(path)
    model.trainable = False
    return model


class Extractor():
    def __init__(self, name, flip=False, resize=True):
        if name == 'nas':
            from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
            self.model = NASNetLarge(include_top=False, weights='imagenet', pooling='avg')
            self.size = 331
            self.f_len= 4032
        if name == 'res':
            from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
            self.model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
            self.size = 224
            self.f_len= 2048
        self.proc = preprocess_input
        self.model.trainable = False
        self.flip = flip
        self.resize = resize

    def get_feature(self, img):
        if self.resize:
            img = cv2.resize(img, (self.size, self.size), cv2.INTER_AREA)
        img = np.expand_dims(img,axis=0)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = self.proc(img)
        if self.flip:
            img_fl = tf.image.flip_left_right(img)
            img = tf.concat([img, img_fl],axis=0)
        f1 = self.model(img)
        f1 = tf.math.l2_normalize(f1).numpy()
        return f1





def tflite_convert_saved_model(in_path, out_path):
    phys_gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(phys_gpu, True)
    converter = tf.lite.TFLiteConverter.from_saved_model(in_path)
    tflite_model = converter.convert()
    tf.io.gfile.GFile(out_path, 'wb').write(tflite_model)

def tflite_convert_keras_model(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tf.io.gfile.GFile(out_path, 'wb').write(tflite_model)

def tflite_convert_pb_file(pb_path, out_path, in_array, out_array, input_dict):
    # in_array   = ["image_tensor"]
    # out_array  = ["num_detections","detection_scores","detection_boxes","detection_classes"]
    # input_dict = {"image_tensor" : [1, 112, 112, 3]}
    tf.lite.TFLiteConverter.inference_input_type = tf.int8
    converter = tf.lite.TFLiteConverter.from_frozen_graph(pb_path, in_array, out_array, input_dict)
    tflite_model = converter.convert()
    tf.io.gfile.GFile(out_path, 'wb').write(tflite_model)


def resnet_preproc(img):
    img = img[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    return img

def mobnet_preproc(img):
    img /= 127.5
    img -= 1.
    return img


# import tflite_runtime.interpreter as tflite

class TFLModel():
    def __init__(self, preproc=(lambda x:x)):
        self.preproc = preproc 

    def load(self, model_path):
        self.model = tf.lite.Interpreter(model_path=model_path)
        self.idetail = self.model.get_input_details()
        self.odetail = self.model.get_output_details()
        self.model.allocate_tensors()

    def forward(self, x):
        # x = np.random.random_sample(size=input_details[0]['shape']).astype(np.float32)
        x = self.preproc(x)
        self.model.set_tensor(self.idetail[0]['index'], x)
        self.model.invoke()
        y = model.get_tensor(self.odetail[0]['index'])
        return y


# assets, variables, saved_model.pb
class FrozenModel():
    def __init__(self, preproc=(lambda x:x)):
        self.preproc = preproc 

    def load(self, model_path):
        self.model = tf.saved_model.load(model_path)
        self.model = self.model.signatures['serving_default']

    def forward(self, x):
        y = self.preproc(x)
        y = self.model(y)
        keys = [*y]
        keys.sort()
        out = []
        for k in keys:
            out.append(y[k][0].numpy())
        if len(out) == 1:
            return out[0]
        return out

class TF1Detector():
    def __init__(self, preproc=(lambda x:x)):
        self.preproc = preproc
        self.in_name = 'image_tensor:0'
        self.out_name = ['detection_boxes:0', 'detection_classes:0', 'detection_scores:0', 'num_detections:0']

    def load(self, model_path):
        graph_file = open(model_path, 'rb')
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(graph_file.read())
        def func():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(func, [])
        graph = wrapped_import.graph

        self.model = wrapped_import.prune(
            tf.nest.map_structure(graph.as_graph_element, self.in_name),
            tf.nest.map_structure(graph.as_graph_element, self.out_name))

    def forward(self, x):
        y = self.preproc(x)
        y = self.model(y)
        out = []
        for i in y:
            out.append(i.numpy()[0])
        return out



