import itertools
import math
import json
import cv2
import torch
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from camera import LoadStreams, LoadImages
from utils.general import non_max_suppression, check_imshow, scale_coords
from yolov5 import Darknet
from threading import Thread

with open('yolov5_config.json', 'r', encoding='utf8') as fp:
    opt = json.load(fp)
    print('[INFO] YOLOv5 Config:', opt)
darknet = Darknet(opt)
if darknet.webcam:
    # cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(darknet.source, img_size=opt["imgsz"], stride=darknet.stride)
else:
    dataset = LoadImages(darknet.source, img_size=opt["imgsz"], stride=darknet.stride)
#########################################################################################################

height = 10  # 高度
fov_width = 120  # 水平视野范围
fov_height = 100  # 垂直视野范围
resolution = (1600, 1200)  # 相机分辨率为1600x1200


def calculate_drone_area(height, fov_width, fov_height, resolution):
    # 计算无人机视野范围
    fov_width_m = 2 * height * math.tan(math.radians(fov_width / 2))
    fov_height_m = 2 * height * math.tan(math.radians(fov_height / 2))
    # 计算每像素代表的实际距离
    pixel_size_m = fov_width_m / resolution[0]
    # 计算无人机检测面积
    drone_area = fov_width_m * fov_height_m / (height * height)
    return drone_area


########################################################################################################
num = 0
state = None
arr = []


def distancing(people_coords, img, dist_thres_lim=(70, 250)):
    already_red = dict()
    centers = []
    for i in people_coords:
        centers.append(((int(i[2]) + int(i[0])) // 2, (int(i[3]) + int(i[1])) // 2))
    for j in centers:
        already_red[j] = 0
    x_combs = list(itertools.combinations(people_coords, 2))
    radius = 10
    thickness = 3
    for x in x_combs:
        xyxy1, xyxy2 = x[0], x[1]
        cntr1 = ((int(xyxy1[2]) + int(xyxy1[0])) // 2, (int(xyxy1[3]) + int(xyxy1[1])) // 2)
        cntr2 = ((int(xyxy2[2]) + int(xyxy2[0])) // 2, (int(xyxy2[3]) + int(xyxy2[1])) // 2)
        dist = ((cntr2[0] - cntr1[0]) ** 2 + (cntr2[1] - cntr1[1]) ** 2) ** 0.5
        if dist < dist_thres_lim[0]:
            color = (0, 255, 255)
            label = "Attention"
            already_red[cntr1] = 1
            already_red[cntr2] = 1
            cv2.line(img, cntr1, cntr2, color, thickness)
            cv2.circle(img, cntr1, radius, color, -1)
            cv2.circle(img, cntr2, radius, color, -1)
            tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
            for xy in x:
                c1, c2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                tf = max(tl - 1, 1)
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)


@csrf_exempt
def acquire():
    global num, state, arr
    for path, img, img0s, vid_cap in dataset:
        img = darknet.preprocess(img)
        drone_area = calculate_drone_area(height, fov_width, fov_height, resolution)  # 总面积
        pred = darknet.model(img, augment=darknet.opt["augment"])[0]
        pred = pred.float()
        pred = non_max_suppression(pred, darknet.opt["conf_thres"], darknet.opt["iou_thres"])
        num = len(pred[0])
        if num > int(drone_area * 4):
            state = '中'
        elif num > int(drone_area * 6):
            state = '高'
        else:
            state = '低'


Thread(target=acquire).start()  # 开启线程，实时获取数据


def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'


@csrf_exempt
def page(request):
    context = {
        'num': num,
        'state': state
    }
    if is_ajax(request=request):
        return JsonResponse(context)
    else:
        return render(request, 'index.html', context)


# 视屏流传输
@csrf_exempt
def showVideo(dataset):
    global n, c
    view_img = check_imshow()
    for path, img, img0s, vid_cap in dataset:
        img = darknet.preprocess(img)
        pred = darknet.model(img, augment=darknet.opt["augment"])[0]  # 0.22s
        pred = pred.float()
        pred = non_max_suppression(pred, darknet.opt["conf_thres"], darknet.opt["iou_thres"])
        people_coords = []
        pred_boxes = []
        for i, det in enumerate(pred):
            if darknet.webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, img0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', img0s, getattr(dataset, 'frame', 0)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {darknet.names[int(c)]}{'s' * (n > 1)}, "
                # Write results
                for *xyxy, conf, cls_id in det:
                    lbl = darknet.names[int(cls_id)]
                    if darknet.names[int(cls_id)] == 'person':
                        xyxy = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                        score = round(conf.tolist(), 3)
                        label = "{}: {}".format(lbl, score)
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        pred_boxes.append((x1, y1, x2, y2, lbl, score))
                        if view_img:
                            label = '%s %.2f' % (darknet.names[int(cls_id)], conf)
                            if label is not None:
                                if (label.split())[0] == 'person':
                                    people_coords.append(xyxy)
                                    # plot_one_box(xyxy, im0, line_thickness=3)
                                    darknet.plot_one_box(xyxy, im0, color=(255, 0, 0), label=label)
            distancing(people_coords, im0, dist_thres_lim=(70, 250))

            if view_img:
                if darknet.names[int(cls_id)] == 'person':
                    cv2.putText(im0, f"{n}{darknet.names[int(c)]}{'s' * (n > 1)}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                    (0, 0, 255), 2)
            #
            #    cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond
            frame = cv2.imencode('.jpg', im0)[1].tobytes()
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'


@csrf_exempt
def video_feed(request, arg_path):
    response = StreamingHttpResponse(showVideo(dataset), content_type='multipart/x-mixed-replace; boundary=frame')
    response['X-My-Header'] = 'Some text data'
    return response
