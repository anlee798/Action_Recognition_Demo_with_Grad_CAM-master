from textwrap import indent
import torch
from torch import nn
import numpy as np
import cv2
torch.backends.cudnn.benchmark = True
import time
from models import shufflenetv2hs
from models import C
from models import shufflevit
import datetime
from torch.autograd import Variable
# python inference_gpu.py
cur_model = shufflevit
is_Rep = False
resume_path = './weights/ucf101_shufflevit_1.0x_RGB_16_best.pth'

Resize_Height = 240
Resize_Width = 320
PerFrameNum = 16

def center_crop(frame):
    frame = frame[8:232,48:272 :]
    return np.array(frame).astype(np.uint8)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    with open('./dataloaders/ucf_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    train_model = cur_model.create_MobileNet(num_classes=101)
    train_model = train_model.cuda()
    train_model = nn.DataParallel(train_model, device_ids=None)
    checkpoint = torch.load(resume_path)
    train_model.load_state_dict(checkpoint['state_dict'])
    
    train_model.to(device)
    train_model.eval()
    flag = True
    all_time = 0
    all_fps = 0
    index_num = 0
    with torch.no_grad():
        if is_Rep:
            rep_model = cur_model.reparameterize_model(train_model).to(device)
        else:
            rep_model = train_model
        #rep_model = HS.reparameterize_model(train_model).to(device)
        # read video
        video = './video_list/kiss.avi'
        cap = cv2.VideoCapture(video)
        retaining = True
        clip = []
        while retaining:
            retaining, frame = cap.read()
            start_time = time.time()
            startTime = datetime.datetime.now()
            if not retaining and frame is None:
                continue
            tmp_ = center_crop(cv2.resize(frame, (Resize_Width, Resize_Height)))#(171,128)
            tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]]) # python inference.py  
            clip.append(tmp)
            if len(clip) == PerFrameNum:
                #inputs = Variable(torch.randn(16, 224, 224, 3))

                inputs = np.array(clip).astype(np.float32) #(16, 224, 224, 3)
                inputs = np.expand_dims(inputs, axis=0)  #(1, 16, 224, 224, 3)
                inputs = np.transpose(inputs, (0, 4, 1, 2, 3)) #(1, 3, 16, 224, 224)
                inputs = torch.from_numpy(inputs)  #torch.Size([1, 3, 16, 224, 224])
                inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device) #torch.Size([1, 3, 16, 224, 224])
                with torch.no_grad():
                    outputs = rep_model.forward(inputs)

                probs = torch.nn.Softmax(dim=1)(outputs)
                label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
                print(label)
                end_time = time.time()
                endTime = datetime.datetime.now()
                ms = ((endTime -startTime ).seconds * 1000 + (endTime -startTime ).microseconds / 1000)
                FPS = 1/(end_time-start_time)
                cv2.putText(frame, 'FPS:'+str(FPS), (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                clip.pop(0)
                if flag:
                    flag = False
                    continue
                all_time += ms
                all_fps += FPS
                index_num += 1
                print("FPS:",FPS)
            cv2.imshow('result', frame)
            cv2.waitKey(30)

        cap.release()
        cv2.destroyAllWindows()

        print("平均推理时间为：",all_time/index_num)
        print("平均推理帧率为：",all_fps/index_num)

# python inference_gpu.py
if __name__ == '__main__':
    main()