# Dronedetection
基于无人机的智能防踩踏监控系统，使用yolov5，旨在通过无人机监控，实现对公共场所人员密集度 的实时监测和预警，从而预防和避免踩踏事故的发生，提高公共安全保障水平。 
我们的前端页面主要显示检测的视屏画面、检测到的总人数、人员密集度，同时视屏画面能够显示密集提示。前端页面主要使用 HTML、CSS、JavaScript 开发，并通过 Ajax
实时更新数据。我们的后端框架采用 Django，主要用于构建 web，传输视频流。在我们进入前端网页中，输入地址，就会调用后端接口，开始进行目标检测并返回视频流，后端进行人员密
集度的判断的信息将会转换成 json 数据，前端会获取 json 数据并且显示。
![image](https://github.com/ML-aChun/yolov5-Dronedetection/assets/94532351/6ec2225b-0a72-4b69-8c59-8aa0880fb685)


# 面积检测
通过无人机的高度传感器能够获取高度，通过无人机摄像头的相机分辨率、垂直视野范围、水平视野范围即可计算出检测面积
![image](https://github.com/ML-aChun/yolov5-Dronedetection/assets/94532351/ef911674-539a-43fd-b6a9-f4286b1d9737)


