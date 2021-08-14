import os, sys, time, math, random, shutil
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageSequence
import matplotlib.pyplot as plt
import scipy

# sys.path.append('/media/anhuynh/DATA/03_task/common_lib')


def rand_str(length=10):
    letters = 'abcdefghijkmnopqrstuvxyzABCDEFGHIJKMNOPQRSTUVXYZ0123456789'
    rand_str = ''
    for i in range(length):
        rand_str+=random.choice(letters)
    return rand_str

def rand_name(dir_path):
    name = rand_str()
    path = os.path.join(dir_path, name)
    while os.path.exists(path):
        name = rand_str()
        path = os.path.join(dir_path, name)
    return (path, name)

def rand_sleep(min_time=0.1):
    rand_lst = [i for i in range(1,10)]
    t = min_time + min_time/random.choice(rand_lst)
    time.sleep(t)

def angle_2_vec(x1,y1,x2,y2):
    angle = np.arctan2( x1*y2-y1*x2, x1*x2+y1*y2 )
    angle = np.degrees(angle)
    return angle

def distance(x1,y1,x2,y2):
    w = x2 - x1
    h = y2 - y1
    d = math.sqrt(w*w+h*h)
    return d

def rotate_img(img, center, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rot = cv2.warpAffine(img, M, (h, w))
    return img_rot

def crop_align_face(img, lm):
    # Calculate auxiliary vectors.
    eye_left     = lm[0]
    eye_right    = lm[1]
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm[3]
    mouth_right  = lm[4]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle. flipud: flip arr up to down. hypot: canh huyen
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    # quad = [top_left, bot_left, bot_right, top_right]
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    quad = (quad + 0.5).astype(np.int)
    qsize = int(np.hypot(*x) * 2)

    x1, y1, x2, y2, x3, y3, x4, y4 = quad.flatten()

    # Center of rect
    cx = int((x1+x2+x3+x4)/4 + 0.5)
    cy = int((y1+y2+y3+y4)/4 + 0.5)

    # Rotating angle
    a1, b1 = (x1-x2, y2-y1)
    a2, b2 = (0., 1.)
    angle = angle_2_vec(a1, b1, a2, b2)

    
    # max distance from center to 4 corners of img
    img_h, img_w = img.shape[:2]
    d1 = distance(cx,cy,0,0)
    d2 = distance(cx,cy,0,img_h)
    d3 = distance(cx,cy,img_w,img_h)
    d4 = distance(cx,cy,img_w,0)
    max_d = int(max(d1,d2,d3,d4))

    # pad width of up, down, left, right. Center is crop rect
    u_pad = max_d - cy
    d_pad = max_d - (img_h - cy)
    l_pad = max_d - cx
    r_pad = max_d - (img_w - cx)

    new_cx = max_d
    new_cy = max_d
    new_center = (new_cx,new_cy)

    # padding
    img_pad = np.pad(img,((u_pad,d_pad),(l_pad,r_pad),(0,0)))

    # rotate img
    img_rot = rotate_img(img_pad, new_center, angle)

    # crop face rect
    crp_x1 = new_cx - qsize//2
    crp_y1 = new_cy - qsize//2
    crp_x2 = crp_x1 + qsize
    crp_y2 = crp_y1 + qsize
    ali_face = img_rot[crp_y1:crp_y2,crp_x1:crp_x2]

    crp_box = np.array([crp_x1,crp_y1,crp_x2,crp_y2])
    img_x1  = l_pad
    img_y1  = u_pad
    img_x2  = l_pad+img_w
    img_y2  = u_pad+img_h
    img_box = np.array([img_x1,img_y1,img_x2,img_y2])
    # new_center = np.array(new_center)

    return img_rot, ali_face, crp_box, img_box, new_center, angle

def crop_align_face2(pad_img, lm, max_d):
    # Calculate auxiliary vectors.
    eye_left     = lm[0]
    eye_right    = lm[1]
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm[3]
    mouth_right  = lm[4]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle. flipud: flip arr up to down. hypot: canh huyen
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    # quad = [top_left, bot_left, bot_right, top_right]
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    quad = (quad + 0.5).astype(np.int)
    qsize = int(np.hypot(*x) * 2)

    x1, y1, x2, y2, x3, y3, x4, y4 = quad.flatten()

    x1 = x1 + max_d
    y1 = y1 + max_d
    x2 = x2 + max_d
    y2 = y2 + max_d
    x3 = x3 + max_d
    y3 = y3 + max_d
    x4 = x4 + max_d
    y4 = y4 + max_d

    # Center of rect
    cx = int((x1+x2+x3+x4)/4 + 0.5)
    cy = int((y1+y2+y3+y4)/4 + 0.5)
    center = (cx,cy)

    # Rotating angle
    a1, b1 = (x1-x2, y2-y1)
    a2, b2 = (0., 1.)
    angle = angle_2_vec(a1, b1, a2, b2)

    # rotate img
    face_rot = rotate_img(pad_img, center, angle)

    # crop face rect
    crp_x1 = cx - qsize//2
    crp_y1 = cy - qsize//2
    crp_x2 = crp_x1 + qsize
    crp_y2 = crp_y1 + qsize
    ali_face = face_rot[crp_y1:crp_y2,crp_x1:crp_x2]

    crp_box = np.array([crp_x1,crp_y1,crp_x2,crp_y2])

    return face_rot, ali_face, crp_box, center, angle

def crop_align_face3(img, lm):
    # Calculate auxiliary vectors.
    eye_left     = lm[0]
    eye_right    = lm[1]
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm[3]
    mouth_right  = lm[4]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    x1 = quad[0,0]
    y1 = quad[0,1]
    x2 = quad[1,0]
    y2 = quad[1,1]
    w  = abs(x1-x2)
    h  = abs(y1-y2)

    transform_size = int(math.sqrt(w*w+h*h))
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)

    return img



def draw_box(image, box, color=(0,255,0),thickness=2):
    x, y ,w, h = box
    x_end = x+w
    y_end = y+h
    cv2.rectangle(image, pt1=(x,y), pt2=(x_end,y_end), color=color, thickness=thickness)

def draw_text(image, text, pos=(20,20), fontScale=0.7, color=(240, 31, 31)):
    cv2.putText(image, text, org=pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=fontScale, color=(0,0,0), thickness=4, lineType=cv2.LINE_AA)
    cv2.putText(image, text, org=pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=fontScale, color=color, thickness=2, lineType=cv2.LINE_AA)

def make_vid(in_video_path, out_video_path, model, start=0, duration=100000):
    cap = cv2.VideoCapture(in_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fourcc  = cv2.VideoWriter_fourcc('M','J','P','G')
    # fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc  = cv2.VideoWriter_fourcc(*'vp80')
    out_vid = cv2.VideoWriter(out_video_path, fourcc, fps, (vid_w,vid_h))

    
    stop = start + (fps*duration)

    # cnt = 0
    # while(True):
    #     ret, frame = cap.read()

    #     if ret == False:
    #         break

    #     if cnt >= start and cnt < stop:
    #         img = model.process_frame(frame)            
    #         out_vid.write(img)
    #         # cv2.imshow('frame',img)
    #         # if cv2.waitKey(1) & 0xFF == ord('q'):
    #         #     break
    #     cnt+=1

    for i in range(vid_l):
        ret, frame = cap.read()

        if i >= start and i < stop:
            img = model.process_frame(frame)            
            out_vid.write(img)
            # cv2.imshow('frame',img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break


    cap.release()
    out_vid.release()
    cv2.destroyAllWindows()

def scale_to(a, newMin, newMax):
    oldRange = (a.max() - a.min())
    newRange = (newMax - newMin)
    scale    = newRange / oldRange
    a_bot_0  = a - a.min()
    newArr   = newMin + a_bot_0 * scale
    return newArr

def reshape_gray(gray):
    gray = np.expand_dims(gray, axis=2)
    gray = np.tile(gray,[1,1,3])
    return gray

def cosine_sim_tf(f1, f2):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import logging
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    
    sim = tf.reduce_sum(f1*f2, axis=1)
    return sim

def json_write(path, py_dict):
    import json
    with open(path, "w") as json_file:  
        json.dump(py_dict, json_file)

def json_read(path):
    import json
    with open(path, "r") as json_file:  
        py_dict = json.load(json_file)
    return py_dict

def str_to_json(text):
    import json
    js = json.loads(text)
    return js

def to_json_str(py_dict):
    import json
    text = json.dumps(py_dict)
    return text

def rm_duplicate(lst):
    rm_lst = []
    for i in lst:
        if i not in rm_lst:
            rm_lst.append(i)
    return rm_lst

def rm_dupl_dict(lst,key):
    rm_lst = []
    for i in lst:
        if not is_in_lst(rm_lst,i,key):
            rm_lst.append(i)
    return rm_lst

def is_in_lst(lst,d,key):
    for i in lst:
        if i[key]==d[key]:
            return True
    return False

def get_file_type(path):
    import magic
    return magic.from_file(path)

def base64_to_img(img_string):
    import base64, io
    img_string = img_string.replace(' ', '+')
    img_string = base64.b64decode(img_string)
    img = np.frombuffer(img_string, np.uint8)
    # img = Image.open(io.BytesIO(img_string)).convert('RGB')
    # img = np.array(img, dtype=np.uint8)
    return img

def img_to_base64(img):
    import base64
    img.tobytes()
    img_string = base64.b64encode(img)
    img_string = str(img_string)[2:-1]
    return img_string

def disable_tf_mem_grow(limit=0):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    import logging
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    phys_gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    if limit > 0:
        vgpu1 = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)
        tf.config.experimental.set_virtual_device_configuration(phys_gpu,[vgpu1])
    else:
        tf.config.experimental.set_memory_growth(phys_gpu, True)

def load_pil(path):
    img_pil = Image.open(path).convert('RGB')
    img = np.array(img_pil)
    return img

def save_pil(path, img, quality=100):
    Image.fromarray(img).save(path, quality=quality)

def save_cv2(path, img):
    cv2.imwrite(path, img)

def load_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def cvt_color(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def lsf(in_dir):
    img_paths = []
    path1s = os.listdir(in_dir)
    for path1 in path1s:
        path = os.path.join(in_dir,path1)
        if os.path.isfile(path):
            img_paths.append(path)
            continue

        path2s = os.listdir(path)
        for path2 in path2s:
            path = os.path.join(in_dir,path1,path2)
            if os.path.isfile(path):
                img_paths.append(path)
                continue

            path3s = os.listdir(path)
            for path3 in path3s:
                path = os.path.join(in_dir,path1,path2,path3)
                if os.path.isfile(path):
                    img_paths.append(path)
                    continue

                path4s = os.listdir(path)
                for path4 in path3s:
                    path = os.path.join(in_dir,path1,path2,path3,path4)
                    if os.path.isfile(path):
                        img_paths.append(path)
                        continue

                    path5s = os.listdir(path)
                    for path5 in path5s:
                        path = os.path.join(in_dir,path1,path2,path3,path4,path5)
                        if os.path.isfile(path):
                            img_paths.append(path)

                        else:
                            print("It's a dir:", path)
    img_paths.sort()
    return img_paths

def ls(path='.',mode=''):
    if mode=='l':
        paths = os.listdir(path)
        paths.sort()
        infos = []
        for p in paths:
            info = {}
            info['name'] = p
            info['path'] = os.path.join(path,p)
            if os.path.isfile(p):
                info['type'] = 'file'
            elif os.path.isdir(p):
                info['type'] = 'dir'
            else:
                info['type'] = 'unk'
            infos.append(info)
        return infos
    else:
        paths = []
        for p in os.listdir(path):
            paths.append(os.path.join(path,p))
        paths.sort()
        return paths

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pwd(path=''):
    if path!='':
        return os.path.abspath(os.path.dirname(path))
    return os.getcwd()

def abspath(path):
    return os.path.abspath(path)

def cp(src, dst):
    if src[-1]=='*':
        pass
    # os.path.exists(path):
    # os.path.isfile()
    # shutil.copyfile(src, dst) # name, name
    shutil.copy(src, dst)     # name, name/dir

def cd(path):
    os.chdir(path)

def rm(path):
    if path[-1]=='*':
        pass
    shutil.rmtree(path)

def exist(path):
    return os.path.exists(path)

def wait_exist(path):
    while not os.path.exists(path):
        time.sleep(.5)
    time.sleep(1)

def mv(src, dst):
    if src[-1]=='*':
        pass
    try:
        shutil.move(src, dst)
        return 0
    except:
        return -1

def cat(f_name):
    with open(f_name, 'r') as file:
        lines = file.readline()
    ret_lines = []
    for line in lines:
        line = line.rstrip("\n")
        ret_lines.append(line)
    return ret_lines

def echo(text, mode, f_name):
    mode = {'>':'w','>>':'a'}[mode]
    with open(f_name, mode) as file:
        file.write(text)

def find_close_parenthesis(text, open_p, close_p, open_pos):
    stack = []
    for i in range(open_pos,len(text)):
        c = text[i]
        if c == open_p:
            stack.append(None)
            if open_p == close_p and len(stack) == 2:
                return i
        elif c == close_p:
            stack.pop()
            if len(stack) == 0:
                return i
    return None

def find_close_tag(text, open_p, close_p, open_pos):
    stack = []
    for i in range(open_pos,len(text)):
        if is_match(text, open_p, i):
            stack.append(None)
        elif is_match(text, close_p, i):
            stack.pop()
            if len(stack) == 0:
                return i
    return None

def is_match(text,sub1,pos1):
    pos2 = pos1 + len(sub1)
    if pos2 > len(text):
        return False
    sub2 = text[pos1:pos2]
    return sub1==sub2

def url_encode(text):
    import urllib
    return urllib.parse.quote(text)
    
def url_decode(text):
    import urllib
    return urllib.parse.unquote(text)

def nms(rects, overlapThresh=0.7):
    if len(rects) == 0:
        return []

    boxes = []
    for rect in rects:
        x,y,w,h = rect
        x2 = x+w
        y2 = y+h
        box = [x,y,x2,y2]
        boxes.append(box)

    rects = np.array(rects).astype("int")
    boxes = np.array(boxes).astype("float")

    # initialize the list of picked indexes 
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return pick

def resize_max_dim(img, max_dim):
    raw_h, raw_w, _ = img.shape
    raw_max = max(raw_h, raw_w)
    if raw_max < max_dim:
        return img
    if raw_max == raw_h:
        scale = max_dim/raw_h
        w = int(raw_w*scale)
        rs_img = cv2.resize(img, (w,max_dim), interpolation=cv2.INTER_LINEAR)
    if raw_max == raw_w:
        scale = max_dim/raw_w
        h = int(raw_h*scale)
        rs_img = cv2.resize(img, (max_dim,h), interpolation=cv2.INTER_LINEAR)
    print("Resize scale:", scale)
    return rs_img

def resize_max_height(img, max_h):
    raw_h, raw_w, _ = img.shape
    raw_max = max(raw_h, raw_w)
    if raw_h < max_h:
        return img
    scale = max_h/raw_h
    w = int(raw_w*scale)
    rs_img = cv2.resize(img, (w,max_h), interpolation=cv2.INTER_LINEAR)
    return rs_img

def inRange(x1,y1,x2,y2,r):
    lSqr = (x1-x2)**2 + (y1-y2)**2
    rSqr = r*r
    return lSqr < rSqr

def inRange2(x1,y1,x2,y2,r):
    l = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return l < r

def distance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def length(line):
    x1,y1,x2,y2 = line
    return distance(x1,y1,x2,y2)

def two_pts_line(x1,y1,x2,y2):
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    return a,b

def two_pts_vector(x1,y1,x2,y2):
    x = x2-x1
    y = y2-y1
    return x,y

def two_vec_angle(vec1, vec2):
    ax, ay = vec1
    bx, by = vec2
    dot = ax*bx + ay*by
    cos_ab = dot / math.sqrt((ax**2+ay**2) * (bx**2+by**2))
    angle = np.arccos(cos_ab)
    return rad_to_deg(angle)

def two_vec_angle2(vec1, vec2):
    ax, ay = vec1
    bx, by = vec2
    angle1 = np.arctan2(ay, ax)
    angle2 = np.arctan2(by, bx)
    angle_1_to_2 = angle2 - angle1
    return rad_to_deg(angle_1_to_2)

def rad_to_deg(rad):
    deg = rad * (180/math.pi)
    return deg

def line_intersection(a1,b1,a2,b2):
    x = (b2-b1)/(a1-a2)
    y = a1*x + b1
    return x,y

def sharpen(img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened

def merge_img(img_lst, axis='w'):
    h,w,_ = img_lst[0].shape
    if axis == 'h':
        sep = np.ones([4, w, 3], dtype=np.uint8) * 255
    else:
        sep = np.ones([h, 4, 3], dtype=np.uint8) * 255

    img_sep_lst = []
    for img in img_lst:
        img_sep_lst.append(img)
        img_sep_lst.append(sep)
    if axis == 'h':
        img  = np.concatenate(img_sep_lst[:-1],axis=0)
    else:
        img  = np.concatenate(img_sep_lst[:-1],axis=1)
    return img



